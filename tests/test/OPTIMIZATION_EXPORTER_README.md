# Hardware Optimization Exporter

## Overview

The Hardware Optimization Exporter is a tool that exports hardware-specific optimization recommendations to deployable configuration files for various ML frameworks. It analyzes performance data and generates ready-to-use implementations, configuration files, and documentation for optimizing ML models on different hardware platforms.

## Key Features

- **Hardware-Specific Recommendations**: Generates tailored optimizations for different hardware (CPU, CUDA, ROCm, MPS, OpenVINO, WebGPU, WebNN)
- **Framework-Specific Templates**: Includes templates for PyTorch, TensorFlow, OpenVINO, WebGPU, and WebNN
- **Multiple Export Formats**: Supports Python, JSON, YAML formats
- **Comprehensive Documentation**: Generates detailed implementation guides and documentation
- **ZIP Archive Support**: Packages all exported files for easy download
- **Interactive UI**: Visualizes optimizations and provides file preview capabilities
- **Batch Export**: Exports multiple optimizations in a single operation

## Components

The exporter consists of several key components:

1. **OptimizationExporter**: Core class that exports recommendations to deployable files
2. **Web Dashboard**: Interactive UI for viewing and exporting optimizations
3. **API Integration**: REST API for programmatic access to export functionality
4. **Command-Line Interface**: CLI for running export operations
5. **Enhanced Visualization UI**: Visual representation of optimization impacts and exports

## Installation

All required components are included in the IPFS Accelerate Python package.

```bash
# Install from source
git clone https://github.com/your-username/ipfs_accelerate_py.git
cd ipfs_accelerate_py
pip install -e .
```

## Usage

### Using the Web Dashboard

1. Start the dashboard:
   ```bash
   cd /path/to/ipfs_accelerate_py/test
   python run_hardware_optimization_dashboard.py
   ```

2. Open the dashboard in your browser: `http://localhost:8050`

3. Navigate to the "Optimization Recommendations" section

4. Select a model and hardware platform, then click "Find Optimizations"

5. View the optimization recommendations

6. Click "Implementation Details" on a recommendation

7. Click "Export Implementation" to export the optimization

8. Configure the export options (framework, format)

9. Click "Export Files" to generate the implementation files

10. View the export results with the enhanced visualization UI

11. Click "Download ZIP Archive" to download all exported files

### Using the Python API

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

# Create ZIP archive of exported files
archive_data = exporter.create_archive(result)
archive_filename = exporter.get_archive_filename(result)

# Save archive to file
with open(f"./optimizations/{archive_filename}", "wb") as f:
    f.write(archive_data.getvalue())

# Export batch optimizations
batch_result = exporter.export_batch_optimizations(
    recommendations_report=report,  # Report with multiple recommendations
    output_dir="./batch_optimizations",
    output_format="all"
)

# Create ZIP archive of batch exports
batch_archive_data = exporter.create_archive(batch_result)
batch_archive_filename = exporter.get_archive_filename(batch_result)

# Save batch archive to file
with open(f"./optimizations/{batch_archive_filename}", "wb") as f:
    f.write(batch_archive_data.getvalue())

# Close exporter
exporter.close()
```

### Using the CLI

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

## Exported Files

The exporter generates the following files for each optimization:

1. **Implementation Files**:
   - `implementation.py`: Python implementation of the optimization
   - Framework-specific implementation files

2. **Configuration Files**:
   - `config.json`: JSON configuration file
   - `config.yaml`: YAML configuration file
   - `config.py`: Python configuration module

3. **Documentation**:
   - `README.md`: Markdown documentation with implementation instructions
   - Performance expectations and hardware requirements

## Enhanced Visualization UI

The exporter includes an enhanced visualization UI that provides:

1. **Performance Impact Visualization**: Visual representation of optimization impacts
2. **File Preview Capabilities**: In-place preview of exported files
3. **Code Copy Functionality**: Quick copy buttons for code snippets
4. **Batch Export Visualization**: Comprehensive visualizations for multi-model exports
5. **Interactive UI Elements**: Accordions, tabs, and responsive elements
6. **ZIP Archive Integration**: Streamlined download of exports

See the [Enhanced Visualization UI Guide](ENHANCED_VISUALIZATION_EXPORT_GUIDE.md) for details.

## Templates

The exporter uses templates to generate framework-specific implementations. These templates are located in:

```
/test/optimization_recommendation/templates/
```

The templates are organized by framework:

- `pytorch/`: PyTorch templates (mixed_precision.py, quantization.py, tensorrt.py)
- `tensorflow/`: TensorFlow templates (mixed_precision.py, quantization.py, tensorrt.py)
- `openvino/`: OpenVINO templates (quantization.py, streams.py)
- `webgpu/`: WebGPU templates (shader_optimization.py, tensor_ops.py)
- `webnn/`: WebNN templates (graph_optimization.py)

## API Endpoints

The exporter provides REST API endpoints for remote access:

1. **POST /api/export-optimization/export**: Export a single optimization
2. **POST /api/export-optimization/batch-export**: Export batch optimizations
3. **GET /api/export-optimization/task/{task_id}**: Get export task status
4. **GET /api/export-optimization/download/{task_id}**: Download ZIP archive

## Next Steps

To get started with the Hardware Optimization Exporter:

1. Review the [Hardware Optimization Guide](HARDWARE_OPTIMIZATION_GUIDE.md)
2. Explore the [Enhanced Visualization UI Guide](ENHANCED_VISUALIZATION_EXPORT_GUIDE.md)
3. Try exporting optimizations for your models using the web dashboard
4. Implement the exported optimizations in your ML pipeline

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

## Contributing

To contribute to the exporter:

1. Add new templates for additional frameworks
2. Enhance existing templates with more optimizations
3. Improve the visualization UI for better user experience
4. Add support for new hardware platforms

## License

This project is licensed under the [MIT License](LICENSE)