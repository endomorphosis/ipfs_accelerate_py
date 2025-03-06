# IPFS Acceleration Testing Guide (Updated March 2025)

This guide provides comprehensive information on testing IPFS acceleration across different hardware platforms and generating reports to analyze performance. The framework now fully integrates with DuckDB for all test results storage.

## Overview

The IPFS acceleration testing framework tests hardware acceleration capabilities across multiple platforms:

- **CUDA**: GPU acceleration using NVIDIA hardware
- **OpenVINO**: Intel hardware optimization
- **CPU**: Standard CPU processing
- **Qualcomm AI Engine**: Mobile device acceleration
- **WebNN**: Browser-based neural network acceleration
- **WebGPU**: Browser-based GPU acceleration (New!)

The framework provides:
- Comprehensive testing of hardware acceleration
- Performance comparison across platforms
- Interactive visualization of results
- Database-first result storage and analysis
- Real-time database integration during testing

## Using the Test Framework

### Running Tests

```bash
# Run tests with default models (small embedding models)
python test_ipfs_accelerate.py

# Run with specific models
python test_ipfs_accelerate.py --models "BAAI/bge-small-en-v1.5,prajjwal1/bert-tiny"

# Test with specific hardware platforms (include Qualcomm)
python test_ipfs_accelerate.py --models "bert-base-uncased" --qualcomm

# Test with WebNN acceleration
python test_ipfs_accelerate.py --models "bert-base-uncased" --webnn

# Test with WebGPU acceleration (New!)
python test_ipfs_accelerate.py --models "bert-base-uncased" --webgpu

# Test with all platform types
python test_ipfs_accelerate.py --models "bert-base-uncased" --qualcomm --webnn --webgpu

# Specify custom endpoint types
python test_ipfs_accelerate.py --models "bert-base-uncased" --endpoints "cuda:0,openvino:0,cpu:0,webnn:0,webgpu:0"

# Force database storage of results (overrides DEPRECATE_JSON_OUTPUT=0)
python test_ipfs_accelerate.py --models "bert-base-uncased" --store-in-db

# Store results ONLY in database (no JSON files)
python test_ipfs_accelerate.py --models "bert-base-uncased" --db-only

# Specify custom database path
python test_ipfs_accelerate.py --models "bert-base-uncased" --db-path ./custom_benchmark.duckdb
```

### Generating Reports

The framework supports four types of reports:

#### 1. General Report

Provides an overview of all test results, including hardware compatibility, model family information, and performance metrics.

```bash
python test_ipfs_accelerate.py --report --format html --output test_report.html
```

#### 2. IPFS Acceleration Report

Focuses specifically on IPFS acceleration results, with detailed information about acceleration types, success rates, and execution times.

```bash
python test_ipfs_accelerate.py --ipfs-acceleration-report --format html --output accel_report.html
```

#### 3. Acceleration Comparison Report

Provides interactive visualizations comparing different hardware acceleration methods, helping you identify the optimal acceleration for your models.

```bash
# Compare all acceleration types across all models
python test_ipfs_accelerate.py --comparison-report --format html

# Compare acceleration for a specific model
python test_ipfs_accelerate.py --comparison-report --model "bert-base-uncased" --format html
```

#### 4. WebGPU Analysis Report (NEW!)

Provides detailed analysis of WebGPU performance characteristics, shader compilation metrics, and browser-specific optimizations.

```bash
# Generate WebGPU analysis report for all browsers
python test_ipfs_accelerate.py --webgpu-analysis --format html

# Generate WebGPU analysis for a specific browser
python test_ipfs_accelerate.py --webgpu-analysis --browser firefox --format html

# Include shader compilation metrics in the analysis
python test_ipfs_accelerate.py --webgpu-analysis --shader-metrics --format html

# Analyze compute shader optimizations
python test_ipfs_accelerate.py --webgpu-analysis --compute-shader-optimization --format html
```

## Report Features

### IPFS Acceleration Report

The IPFS acceleration report provides detailed information about:

- **Success rates** for each acceleration type
- **Execution time** statistics for successful tests
- **Implementation types** (REAL vs MOCK)
- **Model-specific performance** across acceleration types

### Acceleration Comparison Report

The comparison report includes interactive visualizations:

- **Success rate comparison** across acceleration types
- **Performance distribution** with box plots for each acceleration type
- **Model-specific performance** with heatmaps
- **Implementation type breakdown**
- **Key insights and recommendations** based on the data

The report helps you determine:
- Which acceleration type is most reliable for your models
- Which acceleration type provides the best performance
- Which models work best with specific acceleration types
- How to optimize your deployment for different hardware

## Database Storage (Enhanced March 2025)

All test results are now stored in a DuckDB database by default for efficient querying and analysis. JSON output is deprecated in favor of database storage, with test results stored in the database in real-time as they are generated.

### Database Schema

The enhanced database schema includes:

- `ipfs_acceleration_results`: IPFS acceleration test results (real-time storage)
- `test_results`: General test results with extensive metadata
- `performance_results`: Performance metrics with statistical distributions
- `power_metrics`: Power consumption metrics (for mobile/edge devices)
- `hardware_platforms`: Hardware platform information with driver versions
- `models`: Model information with family relationships
- `hardware_compatibility`: Hardware compatibility matrix with scores
- `cross_platform_compatibility`: Cross-platform model compatibility

### Real-Time Storage

The framework now stores test results directly in the database as they are generated:

- Results are saved during the test execution, not just at the end
- Each endpoint test result is immediately stored in the database
- Database access is optimized for concurrent writes
- Run ID tracking ensures results can be properly grouped
- Automatic report generation occurs immediately after test completion

### Key Benefits

The enhanced database integration provides:
- Improved data durability (no lost results even if tests crash)
- Historical tracking of performance trends
- More efficient cross-platform comparisons
- Trend analysis over time with statistical significance
- Customized reporting and querying with SQL
- Reduced disk space usage compared to JSON files (50-80% smaller)
- Faster query performance for complex analyses (5-20x faster)
- Direct integration with visualization tools

## Example: Comprehensive Testing Workflow

For a comprehensive testing workflow, follow these steps:

1. **Test across multiple hardware platforms**:

```bash
# Test with CPU, CUDA, OpenVINO, WebNN, WebGPU and Qualcomm
python test_ipfs_accelerate.py --models "bert-base-uncased,prajjwal1/bert-tiny" --qualcomm --webnn --webgpu --db-only
```

2. **Generate an acceleration comparison report**:

```bash
# Generate HTML report with interactive visualizations
python test_ipfs_accelerate.py --comparison-report --format html --output acceleration_comparison.html
```

3. **Analyze model-specific performance**:

```bash
# Generate comparison for a specific model
python test_ipfs_accelerate.py --comparison-report --model "bert-base-uncased" --format html --output bert_acceleration.html
```

4. **Analyze WebGPU performance across browsers**:

```bash
# Generate WebGPU-specific analysis report
python test_ipfs_accelerate.py --webgpu-analysis --browser firefox --shader-metrics --format html --output webgpu_analysis.html
```

5. **Make data-driven decisions** based on the reports:
   - Choose the optimal hardware acceleration for each model
   - Identify reliability issues with specific acceleration types
   - Optimize deployment based on performance characteristics
   - Select the best browser for WebGPU acceleration

## Advanced Features

### WebGPU Acceleration Testing (NEW!)

The framework now supports testing WebGPU acceleration capabilities:

```bash
# Test with WebGPU acceleration
python test_ipfs_accelerate.py --models "bert-base-uncased" --webgpu

# Test WebGPU with shader metrics monitoring
python test_ipfs_accelerate.py --models "bert-base-uncased" --webgpu --shader-metrics

# Include WebGPU in comparison reports
python test_ipfs_accelerate.py --comparison-report --format html

# Generate detailed WebGPU performance analysis report
python test_ipfs_accelerate.py --webgpu-analysis --format html

# Analyze WebGPU performance for a specific browser
python test_ipfs_accelerate.py --webgpu-analysis --browser firefox --format html

# Analyze shader compilation metrics in WebGPU
python test_ipfs_accelerate.py --webgpu-analysis --shader-metrics --format html

# Analyze compute shader optimizations (best for audio models)
python test_ipfs_accelerate.py --webgpu-analysis --compute-shader-optimization --browser firefox --format html
```

### WebNN Acceleration Testing

The framework supports testing WebNN acceleration capabilities:

```bash
# Test with WebNN acceleration
python test_ipfs_accelerate.py --models "bert-base-uncased" --webnn

# Include WebNN in comparison reports
python test_ipfs_accelerate.py --comparison-report --format html
```

### Qualcomm AI Engine Testing

For testing mobile acceleration with Qualcomm AI Engine:

```bash
# Test with Qualcomm acceleration
python test_ipfs_accelerate.py --models "bert-base-uncased" --qualcomm

# Include power metrics for mobile devices
python test_ipfs_accelerate.py --models "bert-base-uncased" --qualcomm --db-only
```

### Real-Time Database Integration

Test results are now stored directly in the database as they are generated:

```bash
# Store results only in database (no JSON files)
python test_ipfs_accelerate.py --models "bert-base-uncased" --db-only

# Use custom database path
python test_ipfs_accelerate.py --models "bert-base-uncased" --db-path ./custom.duckdb --db-only

# Auto-generate reports immediately after test completion
python test_ipfs_accelerate.py --models "bert-base-uncased" --db-only
```

### Enhanced Visualization

The HTML reports now include interactive Plotly visualizations:
- Bar charts for success rates
- Box plots for performance distribution
- Heatmaps for model-hardware compatibility
- Success rate analysis
- Performance insights with recommendations
- WebGPU performance across browsers
- Shader compilation metrics visualization
- Compute shader optimization analysis
- Hardware performance comparisons
- Browser-specific optimization recommendations

The WebGPU analysis reports provide specialized visualizations:
- Performance comparison across browsers (Chrome/Firefox/Edge/Safari)
- Shader compilation time and count metrics
- Compute shader optimization effectiveness 
- Audio model performance in Firefox vs Chrome (Firefox shows ~20-30% better performance)
- Hardware throughput comparison across acceleration types
- WebGPU shader compilation metrics visualization (NEW!)
- Browser-specific WebGPU performance analysis (NEW!)

## Best Practices

1. **Use database-only storage**: Use `--db-only` to ensure all test results are stored exclusively in the database for maximum efficiency.

2. **Test with multiple hardware platforms**: Compare different acceleration methods (including WebGPU) to find the optimal one for your specific models.

3. **Use HTML format for reports**: HTML reports provide interactive visualizations that make it easier to interpret results.

4. **Compare models of similar size/complexity**: Performance characteristics can vary significantly between model sizes.

5. **Monitor power metrics for mobile deployment**: For edge deployment, pay attention to power efficiency metrics in the reports.

6. **Generate comparison reports regularly**: Monitor performance improvements and hardware compatibility over time.

7. **Test WebGPU across browsers**: Browser implementations of WebGPU can vary significantly in performance - test on Chrome, Firefox, and Edge.

8. **Use real-time database storage**: The enhanced database integration ensures no data is lost even if tests crash.

## Troubleshooting

### Database Connection Issues

If you encounter database connection issues:

```bash
# Specify a custom database path
python test_ipfs_accelerate.py --db-path ./my_benchmark.duckdb --comparison-report

# Check database availability and schema
python scripts/benchmark_db_maintenance.py --check-integrity --db-path ./benchmark_db.duckdb

# Fix database schema issues
python scripts/benchmark_db_maintenance.py --fix-schema --db-path ./benchmark_db.duckdb

# Check database permissions
ls -la ./benchmark_db.duckdb
chmod 644 ./benchmark_db.duckdb
```

### WebGPU/WebNN Testing Issues

If you encounter issues with WebGPU or WebNN testing:

```bash
# Check browser availability for WebGPU
python test_ipfs_accelerate.py --webgpu-analysis --browser-check

# Test with a specific browser
python test_ipfs_accelerate.py --models "bert-base-uncased" --webgpu --browser firefox

# Check shader compatibility
python test_ipfs_accelerate.py --webgpu-analysis --shader-compatibility-check
```

### Missing Hardware Support

If hardware acceleration is not detected:

```bash
# Check available hardware with the hardware detection tool
python hardware_detection.py --verbose

# Run tests with only available hardware
python test_ipfs_accelerate.py --models "bert-base-uncased" --endpoints "cpu:0"

# Force testing with specific hardware (even if not detected)
python test_ipfs_accelerate.py --models "bert-base-uncased" --force-hardware webgpu
```

### Report Generation Failures

If report generation fails:

```bash
# Verify database connectivity
python test_ipfs_accelerate.py --db-path ./benchmark_db.duckdb --report --format json

# Check database schema and tables
python scripts/benchmark_db_maintenance.py --check-schema

# Generate report with debug information
python test_ipfs_accelerate.py --comparison-report --format html --debug

# Try a simpler report format first
python test_ipfs_accelerate.py --report --format markdown
```

### Real-Time Database Storage Issues

If you encounter issues with real-time database storage:

```bash
# Test database write permissions
python scripts/benchmark_db_test.py --write-test

# Fall back to JSON output if needed
python test_ipfs_accelerate.py --models "bert-base-uncased" --no-db

# Use alternative database
python test_ipfs_accelerate.py --models "bert-base-uncased" --db-path ./alternative.duckdb
```

## Conclusion

The IPFS acceleration testing framework provides comprehensive capabilities for testing, comparing, and optimizing hardware acceleration across different platforms. The March 2025 enhancements with real-time DuckDB integration significantly improve the reliability, efficiency, and analysis capabilities of the framework. 

By leveraging the database-first approach, WebGPU testing support, and enhanced visualization tools, you can make data-driven decisions about hardware acceleration for your specific models and use cases.

The framework's real-time database integration ensures test results are stored reliably and efficiently, while the new WebGPU analysis capabilities provide valuable insights into browser-specific performance characteristics. This comprehensive testing framework is essential for optimizing model deployment across the diverse hardware platforms supported by the IPFS Accelerate Python framework.