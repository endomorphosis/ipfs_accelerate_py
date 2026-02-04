# Vision-Text DuckDB Integration

This document describes the DuckDB integration for vision-text models (CLIP, BLIP) in the IPFS Accelerate Python framework. The integration enables tracking model performance and compatibility across hardware platforms (CPU, CUDA, OpenVINO, ROCm, MPS, WebNN, WebGPU).

## Overview

The DuckDB integration for vision-text models:

1. Stores test results in a structured database format
2. Tracks model compatibility across hardware platforms (CPU, CUDA, OpenVINO, ROCm, MPS, WebNN, WebGPU)
3. Generates compatibility matrices and performance reports
4. Creates data visualizations for model performance
5. Integrates with the comprehensive test runner

## Database Schema

Two main tables are used for storing vision-text model data:

### Table: vision_text_results

Stores detailed test results for each model-hardware combination:

| Column | Type | Description |
|--------|------|-------------|
| id | VARCHAR | Unique test result ID |
| model_id | VARCHAR | HuggingFace model ID (e.g., "openai/clip-vit-base-patch32") |
| model_type | VARCHAR | Model type ("clip" or "blip") |
| task | VARCHAR | Model task (e.g., "zero-shot-image-classification") |
| hardware_platform | VARCHAR | Hardware platform (cpu, cuda, openvino) |
| timestamp | TIMESTAMP | Test execution time |
| success | BOOLEAN | Whether the test was successful |
| error_type | VARCHAR | Type of error if test failed |
| avg_inference_time | DOUBLE | Average inference time in seconds |
| memory_usage_mb | DOUBLE | Memory usage in MB |
| results | JSON | Detailed test results |
| metadata | JSON | Test metadata |

### Table: model_compatibility

Stores compatibility information for each model across hardware platforms:

| Column | Type | Description |
|--------|------|-------------|
| model_id | VARCHAR | HuggingFace model ID |
| model_family | VARCHAR | Model family (clip, blip) |
| model_type | VARCHAR | Model type (clip, blip) |
| architecture_type | VARCHAR | Architecture type (vision_text) |
| task | VARCHAR | Model task |
| cpu | BOOLEAN | Compatible with CPU |
| cuda | BOOLEAN | Compatible with CUDA |
| openvino | BOOLEAN | Compatible with OpenVINO |
| rocm | BOOLEAN | Compatible with ROCm |
| mps | BOOLEAN | Compatible with MPS (Apple Silicon) |
| webnn | BOOLEAN | Compatible with WebNN |
| webgpu | BOOLEAN | Compatible with WebGPU |
| last_tested | TIMESTAMP | Last test timestamp |
| last_updated | TIMESTAMP | Last update timestamp |

## Usage

### Running Tests with DuckDB Integration

Use the `--store-results` flag with the comprehensive test runner:

```bash
cd /home/barberb/ipfs_accelerate_py/test

# Test all vision-text models and store results in DuckDB
python run_comprehensive_hf_model_test.py --vision-text --all-hardware --store-results

# Test specific models and store results
python run_comprehensive_hf_model_test.py --models clip,blip --store-results
```

### Working with Test Results

The `vision_text_duckdb_integration.py` script provides tools for working with the stored data:

```bash
# Import test results from a directory
python vision_text_duckdb_integration.py --import-results skills/fixed_tests/collected_results

# List all vision-text models in the database
python vision_text_duckdb_integration.py --list-models

# Generate compatibility matrix
python vision_text_duckdb_integration.py --generate-matrix

# Generate performance report
python vision_text_duckdb_integration.py --performance-report
```

### Generated Reports

Reports are generated in the `reports` directory:

- **JSON Format**: Raw data for programmatic use
- **Markdown Format**: Human-readable compatibility matrix
- **CSV Format**: Performance data for visualization

Example compatibility matrix:

```
# Vision-Text Model Compatibility Matrix

Generated: 2025-03-21 01:09:40

## Model Compatibility

### CLIP Models

| Model | Task | CPU | CUDA | OpenVINO | ROCm | MPS | WebNN | WebGPU | Last Tested |
|-------|------|-----|------|----------|------|-----|-------|--------|-------------|
| openai/clip-vit-base-patch32 | zero-shot-image-classification | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | 2025-03-21 |

### BLIP Models

| Model | Task | CPU | CUDA | OpenVINO | ROCm | MPS | WebNN | WebGPU | Last Tested |
|-------|------|-----|------|----------|------|-----|-------|--------|-------------|
| Salesforce/blip-image-captioning-base | image-to-text | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | 2025-03-21 |
```

## Data Visualization

The `vision_text_visualization.py` script provides interactive visualizations for vision-text model performance data:

```bash
# Create performance comparison visualization
python vision_text_visualization.py --performance-comparison

# Create compatibility heatmap
python vision_text_visualization.py --compatibility-heatmap

# Create time series visualization
python vision_text_visualization.py --time-series

# Create statistical analysis with confidence intervals
python vision_text_visualization.py --statistical-analysis

# Create comprehensive dashboard
python vision_text_visualization.py --dashboard

# Create all visualization types
python vision_text_visualization.py --all

# Use dark theme and open in browser
python vision_text_visualization.py --all --theme dark --browser

# Export visualizations in other formats
python vision_text_visualization.py --performance-comparison --export-format png
```

### Visualization Features

1. **Performance Comparison**:
   - Grouped bar charts comparing inference times across hardware platforms
   - Color-coded by hardware platform
   - Detailed hover information with test counts

2. **Compatibility Heatmap**:
   - Visual matrix of model compatibility across hardware platforms
   - Color-coded by compatibility status (green = compatible, red = incompatible)
   - Grouped by model type (CLIP, BLIP)

3. **Time Series Visualization**:
   - Track performance changes over time
   - Includes interactive range slider for time period selection
   - Line plots with markers for each test data point

4. **Statistical Analysis**:
   - Performance metrics with 95% confidence intervals
   - Error bars showing statistical uncertainty
   - Sample size and detailed statistics on hover

5. **Interactive Dashboard**:
   - Combined view with multiple visualization types
   - Performance comparison, compatibility matrix, and distribution charts
   - Interactive legend and filtering

### Visualization Requirements

The visualization script requires the following Python libraries:

```bash
pip install plotly pandas numpy scipy
```

For static image export (PNG, PDF, SVG), also install:

```bash
pip install kaleido
```

## Advanced Queries

You can directly query the DuckDB database for advanced analytics:

```python
import duckdb

# Connect to database
conn = duckdb.connect('/home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb')

# Get performance statistics for all vision-text models
results = conn.execute("""
  SELECT 
    model_id, 
    model_type,
    hardware_platform, 
    AVG(avg_inference_time) as avg_time,
    COUNT(*) as test_count
  FROM vision_text_results
  WHERE success = TRUE
  GROUP BY model_id, model_type, hardware_platform
  ORDER BY model_type, avg_time
""").fetchdf()

print(results)

# Find models with most compatibility issues
issues = conn.execute("""
  SELECT 
    model_id, 
    model_type,
    COUNT(*) as failed_count,
    STRING_AGG(DISTINCT error_type, ', ') as error_types
  FROM vision_text_results
  WHERE success = FALSE
  GROUP BY model_id, model_type
  ORDER BY failed_count DESC
""").fetchdf()

print(issues)
```

## Integration with Comprehensive Testing

The DuckDB integration is automatically used when running comprehensive tests with the `--store-results` flag. This enables:

1. Automatic storage of test results
2. Building up the compatibility matrix over time
3. Tracking performance changes across hardware platforms
4. Generating reports and visualizations

## Integration with Visualization Dashboard

The vision-text model data can be integrated with the Enhanced Visualization Dashboard for more advanced analytics:

```bash
# Start the enhanced dashboard with vision-text data
python run_enhanced_visualization_dashboard.py --browser --vision-text-focus
```

This provides access to:
- Advanced statistical regression detection
- Correlation analysis between metrics
- Interactive visualizations with confidence intervals
- Comparative analysis across model types and hardware platforms

## Next Steps

1. Expand database schema to store more detailed performance metrics
2. Add support for cross-hardware comparisons
3. Implement visualization dashboards for model performance
4. Integrate with CI/CD pipelines for automated testing
5. Add support for additional model types beyond vision-text