# Hardware Optimization Guide

## Overview

The IPFS Accelerate Python framework includes a comprehensive hardware optimization system that analyzes performance data and provides specific optimization recommendations for different hardware platforms and models. The system includes tools for analyzing performance, generating recommendations, and exporting ready-to-use implementation files.

## Features

- **Performance Analysis**: Analyzes benchmark data to identify optimization opportunities
- **Hardware-Specific Recommendations**: Provides tailored recommendations for different hardware platforms
- **Confidence Scoring**: Rates recommendations based on historical performance data
- **Implementation Details**: Includes detailed implementation instructions
- **Export Capabilities**: Exports recommendations as deployable configuration files
- **Multiple Export Formats**: Supports Python, JSON, YAML formats
- **Framework-Specific Templates**: Includes templates for PyTorch, TensorFlow, ONNX, OpenVINO, WebGPU, and WebNN
- **ZIP Archive Support**: Package all exported files into a downloadable ZIP archive

## Components

### Hardware Optimization Analyzer

The `HardwareOptimizationAnalyzer` analyzes performance data and generates recommendations. It considers:

- Hardware platform characteristics
- Model architecture and family
- Batch size and precision requirements
- Historical performance data

```python
from test.optimization_recommendation.optimization_client import OptimizationClient

# Initialize client
client = OptimizationClient(
    benchmark_db_path="benchmark_db.duckdb",
    predictive_api_url="http://localhost:8080"
)

# Get recommendations
recommendations = client.get_recommendations(
    model_name="bert-base-uncased",
    hardware_platform="cuda",
    batch_size=8,
    current_precision="fp32"
)

# Get specific recommendation details
recommendation = client.get_recommendation_details(
    model_name="bert-base-uncased",
    hardware_platform="cuda",
    recommendation_name="Mixed Precision Training"
)

# Close client
client.close()
```

### Optimization Exporter

The `OptimizationExporter` exports optimization recommendations to deployable configuration files. It supports:

- Multiple output formats (Python, JSON, YAML)
- Framework-specific implementation templates
- Comprehensive documentation generation
- Batch export for multiple recommendations
- ZIP archive creation for easy download

```python
from test.optimization_recommendation.optimization_exporter import OptimizationExporter

# Initialize exporter
exporter = OptimizationExporter(
    output_dir="./optimizations",
    benchmark_db_path="benchmark_db.duckdb",
    api_url="http://localhost:8080"
)

# Export a single optimization
result = exporter.export_optimization(
    model_name="bert-base-uncased",
    hardware_platform="cuda",
    recommendation_name="Mixed Precision Training",
    output_format="all"  # Options: python, json, yaml, script, all
)

print(f"Exported {result['recommendations_exported']} recommendation(s)")
print(f"Files created: {len(result['exported_files'])}")
print(f"Output directory: {result['base_directory']}")

# Export batch optimizations
batch_result = exporter.export_batch_optimizations(
    recommendations_report=report,  # Report with multiple recommendations
    output_dir="./batch_optimizations",
    output_format="all"
)

print(f"Exported {batch_result['exported_count']} optimizations")
print(f"Output directory: {batch_result['output_directory']}")

# Create ZIP archive of exported files
archive_data = exporter.create_archive(result)
archive_filename = exporter.get_archive_filename(result)

# Save archive to file
with open(f"./optimizations/{archive_filename}", "wb") as f:
    f.write(archive_data.getvalue())

print(f"Created archive: {archive_filename}")

# Close exporter
exporter.close()
```

## Web Interface

The system includes a web dashboard for interactive optimization recommendations:

- View optimization recommendations for different models and hardware
- Analyze performance data with charts and visualizations
- Export recommendations as configuration files
- Download exported files as ZIP archives

### Using the Web Dashboard

1. Start the dashboard:
   ```bash
   cd /path/to/ipfs_accelerate_py/test
   python run_hardware_optimization_dashboard.py
   ```

2. Open the dashboard in your browser: `http://localhost:8050`

3. Enter your model and hardware information

4. View optimization recommendations

5. Export and download implementations

## API Integration

The system provides a RESTful API for integration with other tools:

### API Endpoints

- `POST /api/hardware-optimization/recommendations`: Get optimization recommendations
- `GET /api/hardware-optimization/task/{task_id}`: Get task status and results
- `POST /api/export-optimization/export`: Export optimization to files
- `POST /api/export-optimization/batch-export`: Export batch optimizations
- `GET /api/export-optimization/task/{task_id}`: Get export task status
- `GET /api/export-optimization/download/{task_id}`: Download exported files as ZIP archive

### API Example

```python
import requests
import time

# Get optimization recommendations
response = requests.post(
    "http://localhost:8080/api/hardware-optimization/recommendations",
    json={
        "model_name": "bert-base-uncased",
        "hardware_platform": "cuda",
        "batch_size": 8,
        "current_precision": "fp32"
    }
)

task_id = response.json()["task_id"]

# Poll for task completion
while True:
    status_response = requests.get(f"http://localhost:8080/api/hardware-optimization/task/{task_id}")
    status = status_response.json()
    
    if status["status"] != "pending":
        break
    
    time.sleep(1)

# Export recommendation
if status["status"] == "completed":
    recommendations = status["result"]
    
    export_response = requests.post(
        "http://localhost:8080/api/export-optimization/export",
        json={
            "model_name": "bert-base-uncased",
            "hardware_platform": "cuda",
            "recommendation_name": recommendations["recommendations"][0]["name"],
            "output_format": "all"
        }
    )
    
    export_task_id = export_response.json()["task_id"]
    
    # Poll for export task completion
    while True:
        export_status_response = requests.get(
            f"http://localhost:8080/api/export-optimization/task/{export_task_id}"
        )
        export_status = export_status_response.json()
        
        if export_status["status"] != "pending":
            break
        
        time.sleep(1)
    
    # Download exported files as ZIP
    if export_status["status"] == "completed":
        # Download ZIP archive
        download_response = requests.get(
            f"http://localhost:8080/api/export-optimization/download/{export_task_id}",
            stream=True
        )
        
        with open("optimization_exports.zip", "wb") as f:
            for chunk in download_response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("Downloaded optimization exports as ZIP archive")
```

## Command-Line Usage

The system also provides command-line tools for export operations:

```bash
# Export optimization for a model on a specific hardware platform
python -m test.optimization_recommendation.optimization_exporter \
    --model bert-base-uncased \
    --hardware cuda \
    --output-dir ./optimizations \
    --output-format all

# Export optimization and create ZIP archive
python -m test.optimization_recommendation.optimization_exporter \
    --model bert-base-uncased \
    --hardware cuda \
    --output-dir ./optimizations \
    --output-format all \
    --create-archive \
    --archive-path ./archives

# Export optimizations from a batch report
python -m test.optimization_recommendation.optimization_exporter \
    --batch recommendations_report.json \
    --output-dir ./batch_optimizations \
    --output-format all \
    --create-archive
```

## Supported Hardware Platforms

The system supports the following hardware platforms:

- `cpu`: CPU (x86, ARM)
- `cuda`: NVIDIA GPU with CUDA
- `rocm`: AMD GPU with ROCm
- `mps`: Apple Metal Performance Shaders
- `openvino`: Intel Neural Compute with OpenVINO
- `webgpu`: WebGPU for browser-based acceleration
- `webnn`: WebNN for browser-based acceleration

## Recommendation Types

The system provides various types of optimization recommendations:

1. **Mixed Precision Training**: Accelerates training and inference while reducing memory usage
2. **Quantization**: Reduces model size and improves performance with lower precision
3. **TensorRT Integration**: Optimizes models with NVIDIA TensorRT
4. **OpenVINO Streams**: Improves throughput on Intel hardware with multi-stream inference
5. **Shader Precompilation**: Optimizes WebGPU shader loading and compilation
6. **WebNN Graph Optimization**: Optimizes neural network graphs for WebNN
7. **Custom Batch Size Selection**: Optimizes batch size for specific hardware and workloads
8. **Memory Optimization**: Reduces memory usage with techniques like gradient checkpointing
9. **Compiler Optimization**: Uses framework-specific compiler optimizations
10. **Hardware-Specific Kernels**: Uses specialized kernels for specific hardware

## Export Formats

The exporter supports multiple output formats:

- **Python**: Python scripts for direct implementation
- **JSON**: JSON configuration files for framework integration
- **YAML**: YAML configuration files for integration with MLOps tools
- **README**: Markdown documentation with implementation details
- **ZIP Archive**: All generated files packaged as a ZIP archive for easy download

## Customizing Templates

To customize the implementation templates:

1. Create or edit templates in `test/optimization_recommendation/templates/`:
   - `pytorch/`: Templates for PyTorch optimizations
   - `tensorflow/`: Templates for TensorFlow optimizations
   - `openvino/`: Templates for OpenVINO optimizations
   - `webgpu/`: Templates for WebGPU optimizations
   - `webnn/`: Templates for WebNN optimizations

2. Template files should include placeholders that will be replaced by the exporter.

## Troubleshooting

### Common Issues

1. **No recommendations found**:
   - Ensure the benchmark database has sufficient data for the model and hardware
   - Try a more common model or hardware platform

2. **Export fails**:
   - Check that the output directory is writable
   - Verify that the model and hardware platform are correctly specified

3. **Empty ZIP archive**:
   - Ensure that export was successful before creating archive
   - Check that exported files were created in the output directory

## Next Steps

For more information, see the following resources:

- [OPTIMIZATION_API_REFERENCE.md](OPTIMIZATION_API_REFERENCE.md): Complete API reference
- [HARDWARE_BENCHMARKING_GUIDE.md](../data/benchmarks/HARDWARE_BENCHMARKING_GUIDE.md): Guide to hardware benchmarking
- [MODEL_OPTIMIZATION_EXAMPLES.md](../examples/MODEL_OPTIMIZATION_EXAMPLES.md): Examples of optimized models