# Comprehensive Model Compatibility Matrix Guide

**Last Updated:** March 6, 2025

This guide explains how to use the Comprehensive Model Compatibility Matrix, which provides detailed information about cross-platform compatibility for all supported models in the framework.

## Overview

The Comprehensive Model Compatibility Matrix is a powerful tool that visualizes hardware compatibility across all model types in our framework. It enables developers to make informed decisions about hardware selection and deployment strategies based on comprehensive test data stored in our benchmark database.

Key features:
- Cross-platform compatibility status for all 300+ models
- Performance metrics comparison across hardware platforms
- Hardware recommendations by model type and use case
- Interactive filtering and visualization tools
- Automatic weekly updates via CI/CD pipeline
- Export capabilities for integration with other tools

## Technical Architecture

The Compatibility Matrix is built on a robust database architecture that tracks compatibility and performance metrics:

1. **Database Backend**: Uses DuckDB for efficient storage and querying
2. **Data Collection**: Aggregates data from automated tests across all hardware platforms
3. **Templating System**: Uses Jinja2 templates for generating reports in multiple formats
4. **Visualization Layer**: Interactive HTML/JS components for exploring the matrix
5. **CI/CD Integration**: Automatic weekly updates with change tracking

## Compatibility Levels

The matrix uses the following compatibility levels:

| Symbol | Level | Description |
|--------|-------|-------------|
| ✅ | Full Support | Model is fully tested and optimized for this hardware platform |
| ⚠️ | Partial Support | Model works with some limitations (usually performance or memory constraints) |
| 🔶 | Limited Support | Only basic functionality is supported, with significant limitations |
| ❌ | Not Supported | Model is known to be incompatible with this hardware platform |

### Level Criteria

**Full Support (✅)** criteria:
- All model operations function correctly
- Performance is within 10% of reference implementation
- Memory usage is appropriate for the hardware
- All model features and modalities are supported
- Thoroughly tested in production scenarios

**Partial Support (⚠️)** criteria:
- Core model operations function correctly
- Performance is within 30-40% of reference implementation
- May have memory constraints with larger models
- Most features work but may have some limitations
- Tested but with known minor issues or limitations

**Limited Support (🔶)** criteria:
- Basic functionality works but with significant limitations
- Performance is 50% or less compared to reference implementation
- Severe memory constraints limit practical use
- Some features or modalities may not work
- Minimally tested, may have stability issues

**Not Supported (❌)** criteria:
- Model cannot run on this hardware
- Technical limitation prevents implementation
- Hardware incompatible with model architecture
- Memory requirements exceed hardware capabilities
- Failed testing or known critical issues

## Hardware Platforms

The matrix includes compatibility information for the following hardware platforms:

| Platform | Description | Best For |
|----------|-------------|----------|
| CUDA | NVIDIA GPU acceleration | Production servers, high-performance computing |
| ROCm | AMD GPU acceleration | Production servers with AMD hardware |
| MPS | Apple Metal Performance Shaders | Apple Silicon devices, Mac development |
| OpenVINO | Intel acceleration | Intel CPUs/GPUs, edge deployment |
| Qualcomm | Qualcomm AI Engine | Mobile devices, edge computing, power efficiency |
| WebNN | Web Neural Network API | Browser-based deployment, cross-browser compatibility |
| WebGPU | Web GPU API | High-performance browser deployment, GPU acceleration |

## Generating the Matrix

To generate the compatibility matrix, use the following command:

```bash
python test/generate_enhanced_compatibility_matrix.py [options]
```

### Options

- `--db-path PATH`: Path to DuckDB database (default: ./benchmark_db.duckdb)
- `--output-dir DIR`: Output directory for matrix files (default: ./docs)
- `--format FORMAT`: Output format (markdown, html, json, all) (default: all)
- `--filter FILTER`: Filter by model family (e.g., 'bert', 'vision')
- `--hardware HARDWARE`: Filter by hardware platforms (comma-separated)
- `--performance`: Include performance metrics
- `--all-models`: Include all models in the database (not just key models)
- `--recommendations`: Include hardware recommendations
- `--debug`: Enable debug output

### Examples

Generate a comprehensive matrix for all key models:
```bash
python test/generate_enhanced_compatibility_matrix.py --performance --recommendations
```

Generate a matrix for text models only:
```bash
python test/generate_enhanced_compatibility_matrix.py --filter text --performance --recommendations
```

Generate a matrix for specific hardware platforms:
```bash
python test/generate_enhanced_compatibility_matrix.py --hardware CUDA,WebGPU,Qualcomm --performance
```

Generate a JSON-only output:
```bash
python test/generate_enhanced_compatibility_matrix.py --format json --all-models
```

## Setting Up the Database

The compatibility matrix requires a properly configured database with the necessary tables. To set up the database schema and populate it with sample data:

```bash
python test/scripts/create_compatibility_tables.py --db-path ./benchmark_db.duckdb --sample-data
```

This will create the following tables:
- `models`: Information about supported models
- `hardware_platforms`: Information about hardware platforms
- `cross_platform_compatibility`: Compatibility status for model-hardware combinations
- `performance_comparison`: Performance metrics for model-hardware combinations
- `hardware_recommendations`: Recommended hardware platforms for different model types

## Output Formats

### Markdown
The Markdown format (`COMPREHENSIVE_MODEL_COMPATIBILITY_MATRIX.md`) provides a static view of the compatibility matrix, suitable for inclusion in documentation or GitHub repositories. Key features:

- Organized by model modality and family
- Detailed compatibility information with notes
- Performance summaries by hardware platform
- Hardware recommendations by model type

Example:
```markdown
| Model | Family | Parameters | CUDA | ROCm | MPS | OpenVINO | Qualcomm | WebNN | WebGPU | Notes |
|-------|--------|------------|------|------|-----|----------|----------|-------|--------|-------|
| bert-base-uncased | BERT | 110M | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | All platforms fully supported |
| llama-7b | LLaMA | 7B | ✅ | ⚠️ | ⚠️ | ⚠️ | 🔶 | ❌ | 🔶 | WebNN not supported, others limited by memory |
```

### HTML
The HTML format (`compatibility_matrix.html`) provides an interactive matrix with:

- Tabbed interface for matrix, performance, and recommendations views
- Filtering by model type, hardware platform, and compatibility level
- Interactive charts for performance comparisons
- Radar charts comparing hardware platforms on different metrics
- Export functionality to CSV and JSON
- Detailed tooltips with implementation notes
- Search functionality for finding specific models

### JSON
The JSON format (`compatibility_matrix.json`) is machine-readable and suitable for integration with other tools and systems. Structure:

```json
{
  "metadata": {
    "generated_date": "2025-03-06 14:30:22",
    "total_models": 317,
    "total_hardware_platforms": 7,
    "hardware_platforms": ["CUDA", "ROCm", "MPS", "OpenVINO", "Qualcomm", "WebNN", "WebGPU"]
  },
  "compatibility_matrix": [
    {
      "model_name": "bert-base-uncased",
      "model_type": "BertModel",
      "model_family": "BERT",
      "modality": "text",
      "parameters_million": 110,
      "CUDA": "✅",
      "CUDA_level": "full",
      "CUDA_notes": "Optimized performance with CUDA acceleration",
      "ROCm": "✅",
      /* Additional hardware platforms and metadata */
    },
    /* Additional models */
  ],
  "performance_data": {
    /* Performance metrics by hardware and model */
  },
  "recommendations": {
    /* Hardware recommendations by model type */
  }
}
```

## Automated Matrix Updates

The compatibility matrix is automatically updated by our CI/CD pipeline. The GitHub Actions workflow runs on the following schedule:

- Weekly on Sunday at midnight UTC
- On demand via manual workflow dispatch

The workflow does the following:
1. Checks out the repository
2. Sets up the Python environment
3. Generates the compatibility matrix in all formats
4. Calculates differences from the previous version
5. Creates a pull request with the updated matrix if changes are detected
6. Uploads the matrix as an artifact for later reference

This ensures that the compatibility matrix always reflects the latest test results and compatibility information.

## Using the Matrix for Hardware Selection

The compatibility matrix is a valuable tool for selecting the appropriate hardware platform for your models. When making hardware decisions, consider:

1. **Compatibility Level**: Choose platforms with "Full Support" (✅) for optimal results
2. **Performance Metrics**: Compare throughput, latency, and memory usage across platforms
3. **Deployment Context**: Consider the deployment context (server, browser, mobile)
4. **Model Size**: Larger models may require platforms with more memory and compute power
5. **Specific Requirements**: Special requirements like power efficiency for mobile
6. **Budget Constraints**: Some platforms offer better price/performance for specific models

### Decision Flow Chart

```
START
  │
  ├─ Is this for production deployment? ──Yes──► Use CUDA/ROCm when possible
  │  └─No
  │
  ├─ Is this for browser-based deployment? ──Yes──► Consider WebGPU/WebNN
  │  └─No
  │
  ├─ Is this for mobile/edge deployment? ──Yes──► Consider Qualcomm/OpenVINO
  │  └─No
  │
  ├─ Does power efficiency matter? ──Yes──► Prioritize MPS/Qualcomm
  │  └─No
  │
  └─ Does memory usage matter? ──Yes──► Check the Memory Usage metrics
     └─No
        │
        └─► Prioritize platforms with highest throughput
```

## Special Considerations by Model Type

### Text Models
- CUDA is recommended for production deployments of medium to large models
- WebGPU shows excellent performance for smaller models in browser environments
- Qualcomm is the best option for mobile deployments with battery constraints
- ROCm provides good performance on AMD hardware for most text models
- Memory usage increases dramatically with model size, particularly for transformer models
- Batch size optimization is critical for throughput on all platforms

### Vision Models
- Vision models generally show excellent cross-platform compatibility
- WebGPU performance is particularly strong for vision models, making it competitive with native hardware
- CPU can be sufficient for many vision tasks with smaller models
- Vision transformers (ViT) scale better across platforms than CNN architectures
- OpenVINO shows strong optimization for vision models on Intel hardware
- Resolution scaling has significant impact on performance across all platforms

### Audio Models
- Firefox WebGPU shows approximately 20% better performance than Chrome for audio models
- CUDA is still the best option for high-throughput audio processing
- Qualcomm offers good performance with excellent power efficiency for mobile
- Audio models benefit significantly from hardware acceleration for FFT operations
- Real-time audio processing requires careful latency management
- WebGPU audio compute shader optimizations significantly improve performance

### Multimodal Models
- Multimodal models are more demanding and perform best on CUDA GPUs
- Web deployment benefits from parallel loading optimizations
- Memory constraints are the primary limitation on mobile platforms
- Model splitting across encoders can improve parallelization
- CLIP-based models show the best cross-platform compatibility
- LLaVA and other LLM-based multimodal models are the most hardware-constrained

## Performance Metrics

The matrix includes the following performance metrics for each model-hardware combination:

1. **Throughput**: Items processed per second (higher is better)
   - For text models: Tokens per second
   - For vision models: Images per second
   - For audio models: Seconds of audio per second (real-time factor)

2. **Latency**: Processing time in milliseconds (lower is better)
   - End-to-end processing time for a single input
   - Includes preprocessing and postprocessing

3. **Memory Usage**: Peak memory usage in MB
   - Includes model weights, activations, and intermediate tensors
   - Critical for mobile and browser deployments

4. **Power Consumption**: Power usage in watts (lower is better)
   - Particularly important for mobile and edge deployments
   - Critical for battery-powered devices

## Troubleshooting Compatibility Issues

If you encounter compatibility issues:

1. Check the compatibility level and notes in the matrix
2. Verify that your hardware meets the minimum requirements
3. Look for model-specific optimizations in the documentation
4. Consider using a different hardware platform if available
5. For partially supported models, check if reducing batch size or model size helps
6. For WebGPU issues, verify browser version and WebGPU support
7. For mobile platforms, check power and thermal management settings
8. Use performance metrics to identify bottlenecks

### Common Compatibility Issues

| Issue | Possible Solutions |
|-------|-------------------|
| Out of memory errors | Reduce batch size, use smaller model variant, enable memory optimizations |
| Slow inference speed | Check hardware compatibility level, enable platform-specific optimizations |
| Model not loading | Verify hardware support for model architecture, check for specialized operators |
| Browser compatibility | Check browser WebGPU/WebNN support, use browser-specific optimizations |
| Mobile device overheating | Enable thermal management, reduce batch size, use power-efficient settings |
| Unexpected results | Verify quantization settings, check for platform-specific precision issues |

## API Integration

For programmatic access to compatibility data, you can use the compatibility matrix API:

```python
from compatibility_matrix import MatrixAPI

# Initialize API with database path
matrix_api = MatrixAPI(db_path="./benchmark_db.duckdb")

# Get compatibility status for a specific model-hardware combination
status = matrix_api.get_compatibility("bert-base-uncased", "WebGPU")
print(f"Compatibility level: {status['level']}")  # e.g., "full"
print(f"Notes: {status['notes']}")  # e.g., "Optimized for browser environments"

# Get recommended hardware for a model
recommendations = matrix_api.get_recommendations("bert-base-uncased")
print(f"Best hardware: {recommendations['best_platform']}")
print(f"Alternative options: {recommendations['alternatives']}")

# Get performance metrics
metrics = matrix_api.get_performance_metrics("bert-base-uncased", "CUDA")
print(f"Throughput: {metrics['throughput']} items/sec")
print(f"Latency: {metrics['latency']} ms")
print(f"Memory usage: {metrics['memory']} MB")
```

## Maintenance and Updates

The compatibility matrix is automatically updated as part of our CI/CD pipeline. Updates include:

- Weekly scheduled updates with the latest compatibility information
- Updates when new models or hardware platforms are added
- Updates when compatibility status changes
- Delta reports showing improvements since previous updates

## Further Resources

- [Hardware Selection Guide](HARDWARE_SELECTION_GUIDE.md) - Detailed guide for selecting the appropriate hardware
- [Hardware Benchmarking Guide](HARDWARE_BENCHMARKING_GUIDE.md) - How to benchmark models on different hardware
- [Web Platform Support Guide](WEB_PLATFORM_SUPPORT_GUIDE.md) - Details on WebNN and WebGPU support
- [Qualcomm Integration Guide](QUALCOMM_INTEGRATION_GUIDE.md) - Specialized guide for mobile deployment
- [Matrix API Documentation](compatibility_matrix_api.md) - Programmatic access to compatibility data
- [Interactive Dashboard](compatibility_dashboard.md) - How to use the interactive dashboard