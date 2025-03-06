# Comprehensive Benchmark Timing Report Guide

**Date: April 2025**  
**Status: Initial Version**

This guide provides information on using the Comprehensive Benchmark Timing Report tool, which generates detailed performance reports for all 13 key model types across all 8 hardware endpoints.

## Overview

The Benchmark Timing Report tool analyzes the performance data stored in the DuckDB database and generates comprehensive reports showing how each model performs on each hardware platform. It includes:

- Detailed latency and throughput measurements
- Performance comparisons across hardware platforms
- Time-series trend analysis
- Optimization recommendations
- Interactive dashboards for data exploration

## Installation Requirements

The tool has the following dependencies:

```bash
# Core dependencies
pip install duckdb pandas matplotlib seaborn

# Optional for interactive dashboard
pip install streamlit
```

## Basic Usage

### Generating Reports

To generate a comprehensive report in HTML format:

```bash
python run_benchmark_timing_report.py --generate
```

To specify an output file and format:

```bash
python run_benchmark_timing_report.py --generate --format markdown --output benchmark_report.md
```

Available formats:
- `html`: HTML report with interactive visualizations
- `markdown` or `md`: Markdown report for GitHub or documentation
- `json`: Machine-readable JSON format for further processing

### Interactive Dashboard

To launch an interactive dashboard for exploring the benchmark data:

```bash
python run_benchmark_timing_report.py --interactive
```

This will start a Streamlit server (default port: 8501) where you can interactively filter and analyze the benchmark data.

### Integration with Benchmark DB Query System

The timing report is also integrated with the benchmark database query system:

```bash
python benchmark_db_query.py --report timing --format html --output timing_report.html
```

## Configuration Options

The tool supports various configuration options:

| Option | Description | Default |
|--------|-------------|---------|
| `--db-path` | Path to benchmark database | BENCHMARK_DB_PATH env var or ./benchmark_db.duckdb |
| `--format` | Output format (html, markdown, json) | html |
| `--output` | Output file path | benchmark_timing_report.<format> |
| `--days` | Days of historical data to include | 30 |
| `--port` | Port for interactive dashboard | 8501 |

## Report Structure

### HTML Report Structure

The HTML report includes the following sections:

1. **Overview**: Introduction to the report
2. **Hardware Platforms**: Description of all hardware endpoints
3. **Performance Visualizations**: 
   - Latency comparison heatmap
   - Throughput comparison heatmap
   - Historical trend charts
4. **Detailed Results by Category**:
   - Text models (BERT, T5, LLAMA, Qwen2)
   - Vision models (ViT, DETR, XCLIP)
   - Audio models (Whisper, Wav2Vec2, CLAP)
   - Multimodal models (CLIP, LLaVA, LLaVA-Next)
5. **Optimization Recommendations**: Suggested optimizations based on performance data
6. **Conclusion**: Summary of findings

### Markdown Report Structure

The Markdown report follows a similar structure but is optimized for plain text viewing in documentation or GitHub.

### JSON Report Structure

The JSON report provides machine-readable data for further processing:

```json
{
  "generated_at": "2025-04-15T10:30:45",
  "report_type": "benchmark_timing",
  "hardware_platforms": { ... },
  "model_descriptions": { ... },
  "results": [
    {
      "model_name": "bert-base-uncased",
      "model_family": "bert",
      "hardware_type": "cuda",
      "batch_size": 16,
      "average_latency_ms": 12.5,
      "throughput_items_per_second": 128.0,
      "memory_peak_mb": 2048.0,
      "created_at": "2025-04-10T15:30:00"
    },
    ...
  ]
}
```

## Interactive Dashboard Features

The interactive dashboard provides:

1. **Model Filtering**: Select specific models to analyze
2. **Hardware Filtering**: Compare specific hardware platforms
3. **Metric Selection**: Switch between latency, throughput, and memory usage
4. **Performance Comparison Charts**: Visual comparison of performance metrics
5. **Raw Data Table**: Detailed view of the underlying data
6. **Performance Analysis**: Best hardware for each model type

## Interpreting the Results

### Latency (Lower is Better)

Latency represents the time it takes for a model to process a single input. Lower values indicate better performance.

### Throughput (Higher is Better)

Throughput measures how many items the model can process per second. Higher values indicate better performance.

### Memory Usage (Context-Dependent)

Memory usage shows the peak memory consumption during inference. Lower values are generally better, but adequate memory is necessary for model functionality.

## Example Optimization Recommendations

The report provides optimization recommendations based on the benchmark results:

- Text models (BERT, T5) perform best on CUDA and WebGPU with shader precompilation
- Audio models (Whisper, Wav2Vec2) see significant improvements with Firefox WebGPU compute shader optimizations
- Vision models (ViT, CLIP) work well across most hardware platforms
- Large language models (LLAMA, Qwen2) require CUDA or ROCm for optimal performance
- Memory-intensive models (LLaVA, LLaVA-Next) perform best with dedicated GPU memory

## Advanced Usage

### Integrating with CI/CD Pipeline

To automatically generate reports as part of your CI/CD pipeline:

```yaml
# GitHub Actions workflow example
jobs:
  benchmark_report:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Generate benchmark timing report
        run: |
          python test/run_benchmark_timing_report.py --generate --format html --output benchmark_report.html
      - name: Upload report
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-report
          path: benchmark_report.html
```

### Programmatic Usage

You can also use the BenchmarkTimingReport class programmatically:

```python
from benchmark_timing_report import BenchmarkTimingReport

# Initialize with database path
report_gen = BenchmarkTimingReport(db_path="./benchmark_db.duckdb")

# Generate HTML report
report_path = report_gen.generate_timing_report(
    output_format="html",
    output_path="custom_report.html",
    days_lookback=60
)

print(f"Report generated: {report_path}")
```

## Troubleshooting

### Common Issues

1. **Database Not Found**: Ensure the database path is correct and accessible
   ```
   Error: Database not found at: ./benchmark_db.duckdb
   ```
   - Solution: Specify the correct path with `--db-path` or set the `BENCHMARK_DB_PATH` environment variable

2. **No Data Available**: No benchmark data found in the database
   ```
   Warning: No benchmark data available
   ```
   - Solution: Run benchmarks first to populate the database

3. **Missing Dependencies**: Required packages are not installed
   ```
   ImportError: No module named 'streamlit'
   ```
   - Solution: Install the required packages (`pip install streamlit`)

## Future Enhancements

The following enhancements are planned for future releases:

1. **Advanced Analytics**: Statistical analysis of performance differences
2. **Hardware Recommendation Engine**: Automated recommendations for optimal hardware
3. **Regression Detection**: Automatic detection of performance regressions
4. **Export to PowerPoint/PDF**: Direct export to presentation formats
5. **Real-time Monitoring**: Live dashboard for continuous performance monitoring

## Conclusion

The Comprehensive Benchmark Timing Report tool provides valuable insights into the performance characteristics of different models across hardware platforms. By analyzing this data, you can make informed decisions about hardware selection, optimization strategies, and model deployment.