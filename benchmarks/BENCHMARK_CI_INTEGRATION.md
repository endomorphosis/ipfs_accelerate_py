# Benchmark Database CI/CD Integration

This document explains the CI/CD integration for the benchmark database system, which automates the process of running benchmarks, storing results in the database, and generating reports.

## Overview

The CI/CD integration provides the following features:

1. **Automated Benchmarking**: Runs benchmarks for specified models and hardware platforms on every push to main branch or via manual trigger
2. **Result Storage**: Automatically stores benchmark results in the DuckDB database
3. **Report Generation**: Creates HTML reports, compatibility matrices, and comparison charts
4. **Historical Tracking**: Maintains a history of benchmark results for trend analysis
5. **Performance Regression Detection**: Compares results with historical data to detect regressions

## Self-Hosted Runner Requirements

**IMPORTANT**: If using self-hosted runners for benchmarks that utilize Docker containers, the runner user must be added to the docker group:

```bash
sudo usermod -aG docker <runner-user>
```

After making this change, restart the runner service to apply the changes:

```bash
sudo systemctl restart actions-runner
```

For comprehensive setup instructions, see [Self-Hosted Runner Setup Guide](../docs/SELF_HOSTED_RUNNER_SETUP.md).

## GitHub Actions Workflow

The benchmark CI/CD is implemented as a GitHub Actions workflow defined in `.github/workflows/benchmark_db_ci.yml`. The workflow consists of four jobs:

1. **setup_database**: Creates a new database instance for the benchmark run
2. **run_benchmarks**: Runs benchmark tests for specified models and hardware
3. **consolidate_results**: Collects and processes all results into a single database
4. **publish_results**: Generates and publishes reports to GitHub Pages

## Triggering the Workflow

The workflow can be triggered in the following ways:

1. **Automatically on Push**: Runs when code is pushed to the main branch
2. **Pull Requests**: Runs on pull requests to the main branch
3. **Manual Trigger**: Can be triggered manually with custom parameters

### Manual Trigger Options

When triggering the workflow manually, you can specify the following parameters:

- **test_model**: Model to benchmark (default: "all")
- **hardware**: Hardware platform to use (default: "cpu", options: "cpu", "cuda", "all")
- **batch_size**: Comma-separated list of batch sizes to test (default: "1,2,4,8")

## Accessing Benchmark Results

Benchmark results are accessible in several ways:

1. **GitHub Pages**: HTML reports are published to GitHub Pages at `https://{organization}.github.io/{repo}/benchmark-reports/`
2. **Artifacts**: All results are stored as artifacts that can be downloaded from the workflow run
3. **Database Files**: The consolidated database is available as an artifact for detailed analysis

## How It Works

### 1. Database Setup

The `setup_database` job:
- Creates a new DuckDB database with the appropriate schema
- Generates a unique run ID for tracking this benchmark run
- Prepares CI metadata for tracking the source of the benchmarks

### 2. Benchmark Execution

The `run_benchmarks` job:
- Runs in a matrix configuration to test multiple models and hardware platforms
- Uses `run_benchmark_with_db.py` to execute benchmarks and store results directly in the database
- Uploads each individual benchmark result as both database updates and JSON files

### 3. Result Consolidation

The `consolidate_results` job:
- Downloads all benchmark results from the previous job
- Uses `ci_benchmark_integrator.py` to process and consolidate the results
- Generates reports using `benchmark_db_query.py`

### 4. Report Publishing

The `publish_results` job:
- Creates an HTML index page linking to all reports
- Publishes the reports to GitHub Pages
- Stores a copy of the database for historical comparisons

## Running Local Benchmarks

To run benchmarks locally with database integration, use the `run_benchmark_with_db.py` script:

```bash
python test/run_benchmark_with_db.py \
  --db ./benchmark_db.duckdb \
  --model bert-base-uncased \
  --hardware cpu \
  --batch-sizes 1,2,4,8 \
  --iterations 20 \
  --warmup 5
```

## Adding New Models or Hardware

To add a new model or hardware platform to the benchmark matrix:

1. Edit the `.github/workflows/benchmark_db_ci.yml` file
2. Update the `matrix` section in the `run_benchmarks` job
3. Add the new model or hardware platform to the list

## Troubleshooting

### Missing Database Schema

If the workflow fails with "Required tables missing from database", ensure the `create_benchmark_schema.py` script is properly set up.

### Hardware Detection

By default, the workflow runs on GitHub-hosted runners which only provide CPU. For CUDA testing, you'll need to use self-hosted runners.

## Integration with Test Framework

The CI/CD system integrates with the existing test framework through:

1. **Direct API Calls**: The benchmark runner calls the database API directly
2. **CI Artifact Processing**: Results from CI runs are automatically processed
3. **Historical Comparison**: New results are compared with historical data

## Performance Metrics

The CI integration includes performance tracking of the benchmark process itself:

- **Execution Time**: Duration of each benchmark run
- **Database Size**: Growth of the database over time
- **Processing Overhead**: Time spent in result processing vs. actual benchmarking

## Implementation Status

The CI/CD integration for the benchmark database is now **100% complete**. All planned features have been implemented:

- ✅ GitHub Actions workflow for automated benchmarking
- ✅ Matrix-based testing of multiple models and hardware platforms
- ✅ Database integration for result storage
- ✅ Consolidated reporting and visualization
- ✅ Historical data tracking for trend analysis
- ✅ Performance regression detection
- ✅ Artifact storage and management
- ✅ GitHub Pages publishing of reports

## Integration with Hardware Model Predictor

The CI/CD system now integrates with the hardware model predictor to enable advanced performance prediction and analysis:

### Performance Prediction Model Training

The CI/CD workflow includes a new step for training performance prediction models based on benchmark results:

```bash
# CI step for training prediction models
python test/model_performance_predictor.py --train --database ./consolidated_db/benchmark.duckdb --output-dir ./models
```

These prediction models are stored as artifacts and can be used to predict performance for new model-hardware combinations.

### Prediction Accuracy Validation

A new job in the workflow validates prediction accuracy by comparing predicted vs. actual performance:

```bash
# Prediction validation job
python test/hardware_model_predictor.py --validate-predictions --database ./consolidated_db/benchmark.duckdb --output prediction_accuracy.json
```

This helps improve the accuracy of performance predictions over time.

### Hardware Selection Verification

The CI workflow now includes verification of hardware selection recommendations:

```bash
# Test hardware selection recommendations
python test/hardware_model_predictor.py --test-recommendations --models bert-base-uncased t5-small gpt2 --output selection_verification.json
```

This ensures that the hardware selection system makes optimal recommendations based on benchmark data.

## Future Enhancements

With the core CI/CD integration complete, future enhancements could include:

1. ✅ **Advanced Regression Analysis**: Time-series performance tracking with regression detection (Implemented March 7, 2025)
2. ✅ **Automated Issue Creation**: GitHub issue creation for detected performance regressions (Implemented March 7, 2025)
3. **Scheduled Benchmarks**: Regular benchmark runs on a schedule (e.g., weekly or monthly)
4. **Cross-Branch Comparison**: Comparing performance between different branches
5. **Pull Request Comments**: Automatically commenting on PRs with benchmark results
6. **Self-hosted Runner Support**: Enhanced support for CUDA and other specialized hardware on self-hosted runners
7. **Slack/Teams Notifications**: Immediate notifications for significant regressions or benchmark completions
8. **Automated Model Retraining**: Periodically retrain prediction models with new benchmark data
9. **Hardware Selection Recommendations**: Include hardware selection recommendations in PR comments
10. **Performance Prediction in CI Reports**: Add predicted performance for untested configurations

## Conclusion

The CI/CD integration provides a complete system for automated benchmark execution, result storage, and reporting. It ensures that benchmark results are consistently collected, stored, and analyzed, providing valuable insights into performance trends and regressions.

With the addition of the hardware model predictor integration, the system now provides automated performance prediction and hardware recommendation capabilities, further enhancing the project's ability to optimize model deployment across various hardware platforms.

As of March 2, 2025, this component has been fully implemented and integrated with the benchmark database system and hardware selection framework, completing Phase 16 of the project's advanced hardware benchmarking and database consolidation efforts.

For more information, see:
- [Hardware Model Predictor Guide](HARDWARE_MODEL_PREDICTOR_GUIDE.md)
- [Hardware Selection Guide](HARDWARE_SELECTION_GUIDE.md)
- [Phase 16 Implementation Update](PHASE16_IMPLEMENTATION_UPDATE.md)