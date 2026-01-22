# Improved End-to-End Testing Framework

This guide covers the enhancements made to the End-to-End Testing Framework implementation.

## New Features

### Database Integration

Results are now stored in DuckDB for better querying and analysis:

```bash
# Store results in both files and database
python generators/runners/end_to_end/run_e2e_tests.py --model bert-base-uncased --hardware cuda --use-db

# Store results only in database
python generators/runners/end_to_end/run_e2e_tests.py --model bert-base-uncased --hardware cuda --use-db --db-only

# Specify custom database path
python generators/runners/end_to_end/run_e2e_tests.py --model bert-base-uncased --hardware cuda --use-db --db-path ./my_benchmark.duckdb
```

Database integration provides several benefits:
- Efficient storage of test results
- Advanced querying capabilities
- Integration with other components of the framework
- Historical trend analysis
- Performance comparison across hardware

### Distributed Testing

Run tests in parallel for faster execution:

```bash
# Use distributed testing with default number of workers
python generators/runners/end_to_end/run_e2e_tests.py --model-family vision --hardware cuda --distributed

# Specify number of worker threads
python generators/runners/end_to_end/run_e2e_tests.py --all-models --priority-hardware --distributed --workers 8
```

Distributed testing provides:
- Parallel execution of tests across multiple threads
- Significant time savings for large test sets
- Efficient resource utilization
- Automatic task distribution and result collection

### Enhanced Tensor Comparison

Improved validation for tensor outputs with statistical comparison:

```bash
# Specify tolerance for tensor comparison
python generators/runners/end_to_end/run_e2e_tests.py --model vit --hardware cuda --tensor-tolerance 0.05

# Use statistical comparison for large tensors
python generators/runners/end_to_end/run_e2e_tests.py --model bert-base-uncased --hardware cuda --tensor-comparison-mode statistical
```

Key improvements:
- Statistical comparison of tensor distributions
- Configurable tolerance levels for different data types
- Support for different comparison modes (exact, statistical)
- Detailed difference reporting for large tensors

### Real vs. Simulated Hardware Detection

Automatic detection of real vs. simulated hardware:

```bash
# Be explicit about real vs. simulated hardware
python generators/runners/end_to_end/run_e2e_tests.py --model bert-base-uncased --hardware cuda,webgpu --simulation-aware
```

Simulation awareness:
- Automatically detects whether hardware is real or simulated
- Records simulation status in test results
- Provides clear warnings when simulating hardware
- Helps distinguish between real and simulated performance

## Integration with Other Systems

### DuckDB Benchmark Database

Test results are now stored in the same DuckDB database used by other components:

```bash
# Set database path via environment variable
export BENCHMARK_DB_PATH=./benchmark_db.duckdb
python generators/runners/end_to_end/run_e2e_tests.py --model bert-base-uncased --hardware cuda --use-db

# Query test results using benchmark_db_query.py
python duckdb_api/core/benchmark_db_query.py --sql "SELECT * FROM test_results WHERE model_name='bert-base-uncased'" --format markdown
```

### Parallel Documentation Generation

Generate documentation in parallel for faster execution:

```bash
# Generate documentation in parallel
python generators/runners/end_to_end/run_e2e_tests.py --model-family text-generation --hardware all --generate-docs --parallel-docs
```

## Example Usage Scenarios

### Testing a New Model on Multiple Hardware Platforms

```bash
# Test a new model on priority hardware
python generators/runners/end_to_end/run_e2e_tests.py --model new-model --priority-hardware --update-expected

# Generate documentation for the model
python generators/runners/end_to_end/run_e2e_tests.py --model new-model --priority-hardware --generate-docs
```

### Comprehensive Testing of All Models

```bash
# Test all models on priority hardware in parallel
python generators/runners/end_to_end/run_e2e_tests.py --all-models --priority-hardware --distributed --workers 8 --use-db
```

### Integration with CI/CD Pipeline

The end-to-end testing framework has been fully integrated with CI/CD through GitHub Actions workflows. This enables automated testing of all models on various hardware platforms, with result collection, analysis, and reporting.

### GitHub Actions Workflow

A complete workflow has been implemented in `.github/workflows/e2e_testing.yml`. Key features include:

1. **Matrix Testing**: Tests multiple model families across different hardware platforms in parallel
2. **Result Collection**: Gathers and analyzes test results from all jobs
3. **Report Generation**: Creates comprehensive HTML and Markdown reports
4. **GitHub Pages Integration**: Automatically deploys reports to GitHub Pages
5. **Manual Trigger**: Supports manual execution with configurable parameters
6. **Database Integration**: Stores results in DuckDB for historical tracking and analysis

Here's a simplified version of the workflow:

```yaml
name: End-to-End Tests

on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  workflow_dispatch:
    inputs:
      model_family:
        description: 'Model family to test'
        type: choice
        options: [text-embedding, text-generation, vision, audio, multimodal, all]

jobs:
  e2e-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        test-scope:
          - name: 'text-embedding'
            model_family: 'text-embedding'
            hardware: 'cpu,cuda'
          - name: 'vision'
            model_family: 'vision'
            hardware: 'cpu,cuda'
    
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Run end-to-end tests
        run: |
          ./run_e2e_ci_tests.sh --model-family ${{ matrix.test-scope.model_family }} --hardware ${{ matrix.test-scope.hardware }} --distributed
      - name: Upload test results
        uses: actions/upload-artifact@v3
        with:
          name: e2e-test-results-${{ matrix.test-scope.name }}
          path: generators/collected_results/summary/

  analyze-results:
    needs: e2e-tests
    runs-on: ubuntu-latest
    steps:
      - name: Download all test results
        uses: actions/download-artifact@v3
      - name: Combine reports
        run: |
          # Combine all test reports
      - name: Deploy to GitHub Pages
        uses: actions/deploy-pages@v2
```

### Running in CI Mode

When running in CI mode, the testing framework provides enhanced functionality:

```bash
# Run the tests in CI mode directly
python generators/runners/end_to_end/run_e2e_tests.py --model-family text-embedding --hardware cpu,cuda --distributed --ci

# Use the wrapper script for easier CI integration
./run_e2e_ci_tests.sh --model-family text-embedding --hardware cpu,cuda
```

The `--ci` flag enables:
- Special formatting for CI environments
- Status badges generation
- GitHub-flavored Markdown reports
- Git commit information in reports
- Enhanced error reporting for easier debugging

### Setting Up CI/CD

To set up the CI/CD pipeline:

1. Ensure the GitHub repository has the correct permissions:
   - Go to **Settings > Actions > General > Workflow permissions**
   - Select **Read and write permissions**
   - Enable **Allow GitHub Actions to create and approve pull requests**

2. Set up GitHub Pages for report deployment:
   - Go to **Settings > Pages**
   - Set **Source** to **GitHub Actions**

3. Verify the workflow configuration:
   - Check `.github/workflows/e2e_testing.yml` is present and configured correctly
   - Make sure the `run_e2e_ci_tests.sh` script is executable

4. Run the workflow manually to verify:
   - Go to the **Actions** tab in your repository
   - Select **End-to-End Tests** workflow
   - Click **Run workflow** and select the parameters

Refer to `.github/workflows/README.md` for additional details on workflow configuration and troubleshooting.

## Troubleshooting

### Database Connection Issues

If you encounter database connection issues:

```bash
# Check database file exists
ls -l $BENCHMARK_DB_PATH

# Initialize database schema if needed
python duckdb_api/schema/creation/create_benchmark_schema.py
```

### Distributed Testing Problems

If distributed testing isn't working correctly:

```bash
# Try with fewer workers
python generators/runners/end_to_end/run_e2e_tests.py --model bert-base-uncased --hardware cuda --distributed --workers 2 --verbose

# Check for race conditions
python generators/runners/end_to_end/run_e2e_tests.py --model bert-base-uncased --hardware cuda --distributed --verbose
```

### Missing Hardware Detection

If hardware detection isn't working:

```bash
# Install required libraries
pip install torch psutil

# Check hardware detection manually
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Technical Implementation

The improvements to the End-to-End Testing Framework include:

1. **Database Integration**:
   - Uses DuckDB for efficient storage and querying
   - Compatible with the existing benchmark database schema
   - Adds simulation status and test metadata

2. **Distributed Testing**:
   - Implements a thread pool for parallel test execution
   - Uses queue-based task distribution
   - Ensures thread-safe result collection

3. **Enhanced Tensor Comparison**:
   - Supports statistical comparison of tensor distributions
   - Configurable tolerance levels
   - Detailed difference reporting

4. **Real vs. Simulated Hardware Detection**:
   - Automatic detection based on available hardware
   - Clear indication in test results and reports
   - Integration with the simulation tracking system

## Future Improvements

Planned future enhancements:

1. **Remote Worker Support**: Distribute tests across multiple machines
2. **Real-Time Results Dashboard**: View test results in real-time
3. **Advanced Analysis Tools**: Compare results across hardware and models
4. **Predictive Test Selection**: Use ML to predict which tests to run
5. **Integration with Predictive Performance System**: Compare actual vs. predicted performance