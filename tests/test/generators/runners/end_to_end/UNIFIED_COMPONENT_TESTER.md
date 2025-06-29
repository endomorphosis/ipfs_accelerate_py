# Unified Component Tester

## Overview

The Unified Component Tester is an enhanced framework for generating and testing skill, test, and benchmark components together for AI models across different hardware platforms. It builds on the existing integrated component testing approach to provide:

1. **Complete Component Integration**: Generates, tests, and evaluates all components together as a cohesive unit
2. **Enhanced Result Organization**: Creates robust expected/collected results structure with historical tracking
3. **Advanced Documentation Generation**: Produces detailed markdown documentation with model-family and hardware-specific content
4. **Comprehensive Testing**: Validates components across all model families and hardware platforms
5. **Template-Driven Development**: Uses a robust template system for maintenance efficiency
6. **Parallel Execution**: Supports multi-worker execution for faster testing
7. **Database Integration**: Stores results in DuckDB with simulation status tracking
8. **Interactive Visualization**: Provides a web-based dashboard for exploring test results and performance metrics
9. **Integrated Reporting**: Combines visualization dashboard and CI/CD reporting tools in a unified system

This implementation directly addresses the prioritized tasks for the improved End-to-End Testing Framework as specified in CLAUDE.md.

## Key Features

- **Joint Component Generation**: Generates skill, test, and benchmark files together using templates
- **Comprehensive Testing**: Tests all components as a unified whole
- **Result Validation**: Compares results with expected outputs using configurable tolerances
- **Documentation Generation**: Creates detailed Markdown documentation for implementations
- **Database Integration**: Stores results in DuckDB for efficient querying and analysis
- **Expected/Collected Organization**: Maintains clear directory structure for expected and actual results
- **Template-Driven Approach**: Uses templates from a centralized database for consistency
- **Parallel Execution**: Distributes tests across multiple worker processes
- **Model Family Support**: Specialized handling for different model families
- **Hardware Platform Coverage**: Comprehensive support for all hardware platforms
- **Test Reporting**: Detailed test reports with success/failure status and metrics
- **CI/CD Integration**: Optimized for continuous integration environments
- **Visualization Dashboard**: Interactive web dashboard for real-time monitoring and analysis of test results

## Usage

### Basic Usage

```bash
# Test a single model on a single hardware platform
python unified_component_tester.py --model bert-base-uncased --hardware cpu

# Test with documentation generation
python unified_component_tester.py --model bert-base-uncased --hardware cpu --generate-docs

# Update expected results
python unified_component_tester.py --model bert-base-uncased --hardware cpu --update-expected

# Enable verbose logging
python unified_component_tester.py --model bert-base-uncased --hardware cpu --verbose
```

### Advanced Usage

```bash
# Test a model on multiple hardware platforms
python unified_component_tester.py --model bert-base-uncased --hardware cpu,cuda,webgpu

# Test all models in a family
python unified_component_tester.py --model-family text-embedding --hardware cpu

# Test all supported models on priority hardware
python unified_component_tester.py --all-models --priority-hardware

# Run with multiple worker processes
python unified_component_tester.py --all-models --priority-hardware --max-workers 4

# Clean up old test results
python unified_component_tester.py --clean-old-results --days 14

# Run a quick test with minimal validation
python unified_component_tester.py --model bert-base-uncased --hardware cpu --quick-test

# Keep temporary files for debugging
python unified_component_tester.py --model bert-base-uncased --hardware cpu --keep-temp

# Set a custom tolerance for numeric comparisons
python unified_component_tester.py --model bert-base-uncased --hardware cpu --tolerance 0.05
```

### CI/CD Integration

For continuous integration environments:

```bash
# Run the CI-optimized tests
python ci_unified_component_test.py
```

### Testing the Tester

For comprehensive testing of the unified component tester itself:

```bash
# Run basic tests
./run_unified_component_tests.sh

# Run with realistic tests (takes longer)
./run_unified_component_tests.sh --realistic
```

## Architecture

The unified component tester is designed with a modular architecture:

1. **UnifiedComponentTester Class**: Core class responsible for orchestrating the testing process
2. **Component Generation**: Creates skill, test, and benchmark files from templates
3. **Test Execution**: Runs tests and captures results
4. **Benchmark Execution**: Runs benchmarks and captures performance metrics
5. **Documentation Generation**: Creates detailed documentation for each implementation
6. **Result Storage**: Stores results in both file system and database
7. **Result Comparison**: Compares actual results with expected results
8. **Visualization Dashboard**: Provides interactive web UI for exploring test results and metrics

### Class Structure

```
UnifiedComponentTester
├── __init__(): Initialize tester with model, hardware, and options
├── _determine_model_family(): Determine model family from model name
├── _get_git_hash(): Get current git commit hash
├── generate_components(): Generate skill, test, and benchmark files
├── _render_template(): Render a template with model and hardware information
├── _get_template(): Get a template from the database or template files
├── run_test(): Run tests for the model on specified hardware
├── run_benchmark(): Run benchmarks for the model on specified hardware
├── generate_documentation(): Generate documentation for the implementation
├── store_results(): Store test, benchmark, and documentation results
├── compare_results(): Compare test results with expected results
├── run_test_with_docs(): Run complete workflow with documentation
└── run(): Run the complete testing workflow
```

### Helper Functions

- `run_unified_test()`: Run a test for a single model/hardware combination
- `run_batch_tests()`: Run tests for multiple model/hardware combinations
- `clean_old_results()`: Clean up old collected results
- `main()`: Parse command line arguments and run tests

## Directory Structure

```
generators/
├── expected_results/        # Expected outputs for regression testing
│   ├── bert-base-uncased/
│   │   ├── cpu/
│   │   │   └── expected_results.json
│   │   └── ...
│   └── ...
├── collected_results/       # Actual test results with timestamps
│   ├── bert-base-uncased/
│   │   ├── cpu/
│   │   │   └── 20250310_120000/
│   │   │       └── results.json
│   │   └── ...
│   └── summary/             # Summary reports from test runs
├── model_documentation/     # Generated documentation
│   ├── bert-base-uncased/
│   │   ├── bert-base-uncased_cpu_docs.md
│   │   └── ...
│   └── ...
└── runners/
    └── end_to_end/          # End-to-end testing framework scripts
        ├── unified_component_tester.py      # Main unified component tester
        ├── test_unified_component_tester.py # Test suite for the tester
        ├── ci_unified_component_test.py     # CI/CD integration tests
        ├── run_unified_component_tests.sh   # Shell script for running tests
        ├── template_validation.py           # Validation and comparison logic
        ├── model_documentation_generator.py # Documentation generator
        ├── visualization_dashboard.py       # Interactive dashboard for results
        ├── test_visualization_dashboard.py  # Test suite for the dashboard
        ├── dashboard_requirements.txt       # Dashboard dependencies
        └── simple_utils.py                  # Utility functions
```

## Component Generation

The tester generates components using a template-driven approach:

1. **Template Selection**: Based on model family and hardware platform
2. **Parameter Substitution**: Fills in model-specific and hardware-specific details
3. **Validation**: Ensures generated components are correct and complete
4. **Output**: Creates skill, test, and benchmark files in a temporary directory

Templates may come from:
- The template database (`template_database.duckdb`)
- The template renderer (`template_renderer.py`)
- Default templates in the tester itself (fallback)

## Testing Process

The unified testing workflow follows these steps:

1. **Component Generation**:
   - Retrieves appropriate templates for model/hardware combination
   - Fills in parameters and generates skill, test, and benchmark files
   - Validates generated files for correctness

2. **Test Execution**:
   - Runs the test file with the skill implementation
   - Captures test results and success/failure status

3. **Benchmark Execution**:
   - Runs the benchmark file with the skill implementation
   - Captures performance metrics (latency, throughput, etc.)

4. **Documentation Generation**:
   - Creates comprehensive Markdown documentation
   - Includes implementation details, API documentation, and results

5. **Result Storage**:
   - Saves results to timestamped directories
   - Updates expected results if requested
   - Stores results in DuckDB database if available

6. **Result Comparison**:
   - Compares actual results with expected results
   - Uses configurable tolerances for numerical values
   - Provides detailed comparison information

## Improvements Over Previous Framework

The unified component tester improves upon the integrated component test runner in several ways:

1. **Robustness**:
   - Better error handling and recovery
   - Improved template fallback mechanisms
   - Enhanced validation of generated components

2. **Efficiency**:
   - Multi-worker support for parallel testing
   - Quick test mode for faster validation
   - Improved temporary file management

3. **Documentation**:
   - More detailed model-family-specific documentation
   - Better hardware-specific optimization information
   - Improved benchmark visualization and analysis

4. **Testing**:
   - Comprehensive test suite for the tester itself
   - CI/CD-specific test script
   - Automated testing script for validation

5. **User Experience**:
   - More command-line options for flexibility
   - Better progress reporting and logging
   - Improved result organization

## Model Families

The tester supports the following model families:

| Family | Description | Examples |
|--------|-------------|----------|
| text-embedding | Text embedding models | BERT, Sentence Transformers |
| text-generation | Text generation models | OPT, T5, Falcon |
| vision | Computer vision models | ViT, DETR, CLIP (vision) |
| audio | Audio processing models | Whisper, Wav2Vec2, CLAP |
| multimodal | Multimodal models | CLIP, LLaVA, FLAVA |

## Hardware Platforms

The tester supports the following hardware platforms:

| Platform | Description |
|----------|-------------|
| cpu | CPU execution |
| cuda | NVIDIA GPU execution with CUDA |
| rocm | AMD GPU execution with ROCm |
| mps | Apple Metal Performance Shaders |
| openvino | Intel OpenVINO acceleration |
| qnn | Qualcomm Neural Networks API |
| webnn | Web Neural Network API |
| webgpu | WebGPU acceleration |

Priority hardware platforms (tested by default) are CPU, CUDA, OpenVINO, and WebGPU.

## Expected Results and Regression Testing

The tester supports regression testing by comparing results with expected baselines:

```bash
# Update expected results
python unified_component_tester.py --model bert-base-uncased --hardware cpu --update-expected

# Compare with expected results
python unified_component_tester.py --model bert-base-uncased --hardware cpu
```

This ensures that changes to the implementation don't unexpectedly affect results. The `--tolerance` parameter allows configuring the acceptable difference between actual and expected results.

## Database Integration

The tester integrates with DuckDB for efficient storage and retrieval of test results:

```bash
# Use a specific database path
python unified_component_tester.py --model bert-base-uncased --hardware cpu --db-path ./benchmark_db.duckdb

# Disable database storage
python unified_component_tester.py --model bert-base-uncased --hardware cpu --no-db
```

The database schema includes tables for:
- Test results with comprehensive metadata
- Hardware capabilities and detection information
- Model information and characteristics
- Performance metrics and comparisons

## Visualization Dashboard and Integrated Reports System

The tester is integrated with both an interactive Visualization Dashboard for exploring test results and performance metrics, and an Integrated Reports System that combines the dashboard with the Enhanced CI/CD Reports Generator.

### Visualization Dashboard

The Visualization Dashboard provides an interactive web interface for exploring and analyzing test results:

```bash
# Start the visualization dashboard with default settings
python visualization_dashboard.py

# Use a custom database path and port
python visualization_dashboard.py --port 8050 --db-path ./benchmark_db.duckdb

# Run in development mode with hot reloading
python visualization_dashboard.py --debug
```

#### Dashboard Features

The dashboard includes five main tabs for comprehensive test result analysis:

1. **Overview Tab**: Provides a high-level summary of test results including success rates, test counts by hardware platform, and model distribution.

2. **Performance Analysis Tab**: Allows detailed exploration of performance metrics for specific models and hardware combinations, with interactive filtering and comparisons.

3. **Hardware Comparison Tab**: Enables side-by-side comparison of different hardware platforms, highlighting optimal hardware for each model type.

4. **Time Series Analysis Tab**: Shows performance trends over time with statistical analysis to identify significant changes and potential regressions.

5. **Simulation Validation Tab**: Validates the accuracy of hardware simulations by comparing metrics between simulated and real hardware.

### Integrated Visualization and Reports System

For enhanced functionality, the framework includes an integrated system that combines the visualization dashboard with the CI/CD reporting tools:

```bash
# Start the dashboard only
python integrated_visualization_reports.py --dashboard

# Generate reports only
python integrated_visualization_reports.py --reports

# Start dashboard and generate reports
python integrated_visualization_reports.py --dashboard --reports

# Specify database path and automatically open browser
python integrated_visualization_reports.py --dashboard --db-path ./benchmark_db.duckdb --open-browser

# Generate specific report types
python integrated_visualization_reports.py --reports --simulation-validation

# Export dashboard visualizations for offline viewing
python integrated_visualization_reports.py --dashboard-export
```

#### Integrated System Benefits

The integrated system provides several advantages:

- **Unified Interface**: Consistent command-line interface for both the dashboard and reports
- **Database Integration**: Ensures both systems use the same database configuration
- **Process Management**: Handles starting, monitoring, and graceful shutdown of the dashboard process
- **Browser Integration**: Can automatically open the dashboard in your web browser
- **Export Capabilities**: Exports dashboard visualizations for offline viewing
- **Report Customization**: Generates specialized reports for different use cases
- **CI/CD Support**: Optimized for continuous integration environments

#### Report Types

The integrated system supports generating various report types:

```bash
# Generate simulation validation report
python integrated_visualization_reports.py --reports --simulation-validation

# Generate cross-hardware comparison report
python integrated_visualization_reports.py --reports --cross-hardware-comparison

# Generate a combined report with multiple analyses
python integrated_visualization_reports.py --reports --combined-report

# Generate historical trend analysis
python integrated_visualization_reports.py --reports --historical --days 30

# Generate CI/CD status badges
python integrated_visualization_reports.py --reports --badge-only

# Generate reports with enhanced visualizations
python integrated_visualization_reports.py --reports --include-visualizations
```

### Installing Dashboard Dependencies

To use the dashboard and integrated reports system, you'll need to install the following dependencies:

```bash
pip install -r dashboard_requirements.txt
```

### Accessing the Dashboard

Once started, the dashboard is accessible in your web browser at `http://localhost:8050` (or the custom port you specified). The dashboard provides real-time monitoring with automatic refreshing at configurable intervals.

For detailed documentation on the visualization dashboard and integrated system, see:
- [VISUALIZATION_DASHBOARD_README.md](./VISUALIZATION_DASHBOARD_README.md) - Comprehensive dashboard guide

## Documentation Generation

The tester can automatically generate detailed documentation for model implementations:

```bash
# Generate documentation for a specific model and hardware
python unified_component_tester.py --model bert-base-uncased --hardware cpu --generate-docs
```

The generated documentation includes:
- Implementation details and architecture
- API documentation with method descriptions
- Usage examples
- Test results and performance metrics
- Hardware-specific optimizations
- Limitations and recommendations

## Best Practices

1. **Update Expected Results After Changes**:
   When making significant changes to model implementations or templates, update the expected results:
   ```bash
   python unified_component_tester.py --model your-model --hardware your-hardware --update-expected
   ```

2. **Generate Documentation**:
   Keep documentation up-to-date when making changes:
   ```bash
   python unified_component_tester.py --model your-model --hardware your-hardware --generate-docs
   ```

3. **Clean Up Old Results**:
   Periodically clean up old collected results to save disk space:
   ```bash
   python unified_component_tester.py --clean-old-results --days 14
   ```

4. **Focus on Templates**:
   When fixing issues, focus on the template generators rather than individual files.
   
5. **Test Across Hardware Platforms**:
   Ensure models work on all necessary hardware platforms:
   ```bash
   python unified_component_tester.py --model your-model --hardware cpu,cuda,openvino,webgpu
   ```

6. **Use Parallel Testing for Large Workloads**:
   When testing many models or hardware platforms, use multiple workers:
   ```bash
   python unified_component_tester.py --all-models --priority-hardware --max-workers 4
   ```

7. **Maintain a Comprehensive Test Suite**:
   Regularly run the tests for the tester itself:
   ```bash
   ./run_unified_component_tests.sh
   ```

## Troubleshooting

1. **Tests Are Failing But Implementation Looks Correct**:
   - Check if expected results need updating: `--update-expected`
   - Examine the differences in test results
   - Adjust the tolerance level if precision differences are expected: `--tolerance 0.05`

2. **Documentation Is Not Generating Correctly**:
   - Check that components are being generated correctly
   - Verify the model and hardware names are correct
   - Check template variable substitution
   - Use the verbose flag to see more details: `--verbose`

3. **Database Integration Issues**:
   - Verify the DuckDB installation with: `pip install duckdb==0.9.2`
   - Check database path permissions
   - Try using `--no-db` to disable database integration temporarily
   - Check database error logs

4. **Template Issues**:
   - Check template rendering parameters
   - Verify template database path
   - Use `--verbose` to see detailed template operations
   - Check if the template database exists and has required templates

5. **Performance Issues**:
   - Reduce worker count if system resources are limited
   - Use quick test mode for faster testing: `--quick-test`
   - Run tests for specific models or hardware platforms instead of all
   - Check system resource usage during testing

## Implementation Status

The Unified Component Tester successfully implements all the required features for the Improved End-to-End Testing Framework as prioritized in CLAUDE.md:

✅ **Generation and testing of all components together for every model**
- Implemented in the `generate_components()` method with robust template handling
- Tests all components as a unified whole with the `run_test()` method
- Validates components work together with comprehensive error checking

✅ **Creation of "expected_results" and "collected_results" folders for verification**
- Creates organized directory structure in the `store_results()` method
- Supports comparison between expected and actual results with `compare_results()`
- Enables regression testing with configurable tolerance

✅ **Markdown documentation of HuggingFace class skills to compare with templates**
- Generates detailed documentation with the `generate_documentation()` method
- Includes model-family-specific content and hardware-specific optimizations
- Supports visualization of benchmark results with charts

✅ **Focus on fixing generators rather than individual test files**
- Uses template-driven approach for all component generation
- Includes robust validation of generated components
- Supports template inheritance and specialization for different model families and hardware

✅ **Template-driven approach for maintenance efficiency**
- Uses centralized template database for all component generation
- Supports model-family and hardware-specific templates
- Implements robust error handling and fallback mechanisms

✅ **Interactive visualization dashboard for test results**
- Provides a comprehensive web-based dashboard with five specialized tabs
- Enables real-time monitoring of test execution and results
- Supports detailed analysis of performance metrics and trends
- Facilitates hardware comparison and simulation validation
- Includes statistical analysis for identifying significant changes

✅ **Integrated visualization and reports system**
- Combines the visualization dashboard with the Enhanced CI/CD Reports Generator
- Provides a unified command-line interface for both systems
- Ensures consistent database access across all components
- Supports specialized report generation for different use cases
- Includes process management for dashboard startup and shutdown
- Features browser auto-opening and export capabilities

## Contact

If you have questions or need support with the unified component tester, please contact the infrastructure team.