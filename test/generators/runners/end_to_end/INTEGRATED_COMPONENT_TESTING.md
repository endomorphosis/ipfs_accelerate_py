# Integrated Component Testing

> **New Feature**: The Visualization Dashboard and Integrated Reports System are now available for exploring test results and performance metrics interactively. See the [Visualization Dashboard README](VISUALIZATION_DASHBOARD_README.md) for comprehensive documentation and the [Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md) for solutions to common issues.

## Overview

The Integrated Component Testing framework is a comprehensive solution for generating and testing model skills, tests, and benchmarks together as coherent units. This approach addresses several key problems in the previous testing workflow:

1. **Component Cohesion**: Ensures all components work together seamlessly
2. **Maintenance Efficiency**: Focuses on fixing generators rather than individual files
3. **Documentation Consistency**: Automatically generates documentation for each implementation
4. **Regression Prevention**: Compares results against expected baselines
5. **Result Persistence**: Stores comprehensive test results for analysis

## Key Features

- **Joint Component Generation**: Creates skill, test, and benchmark files together using templates
- **Comprehensive Testing**: Tests all components as a unified whole
- **Result Validation**: Compares results with expected outputs using configurable tolerances
- **Documentation Generation**: Creates detailed Markdown documentation for implementations
- **Database Integration**: Stores results in DuckDB for efficient querying and analysis
- **Expected/Collected Organization**: Maintains clear directory structure for expected and actual results
- **Template-Driven Approach**: Uses templates from a centralized database for consistency

## Usage

### Basic Usage

```bash
# Test a single model on a single hardware platform
python integrated_component_test_runner.py --model bert-base-uncased --hardware cpu

# Test with documentation generation
python integrated_component_test_runner.py --model bert-base-uncased --hardware cpu --generate-docs

# Update expected results
python integrated_component_test_runner.py --model bert-base-uncased --hardware cpu --update-expected

# Enable verbose logging
python integrated_component_test_runner.py --model bert-base-uncased --hardware cpu --verbose
```

### Advanced Usage

```bash
# Test a model on multiple hardware platforms
python integrated_component_test_runner.py --model bert-base-uncased --hardware cpu,cuda,webgpu

# Test all models in a family
python integrated_component_test_runner.py --model-family text-embedding --hardware cpu

# Test all supported models on priority hardware
python integrated_component_test_runner.py --all-models --priority-hardware

# Clean up old test results
python integrated_component_test_runner.py --clean-old-results --days 14
```

## Directory Structure

```
generators/
├── expected_results/        # Expected outputs for regression testing
│   ├── bert-base-uncased/
│   │   ├── cpu/
│   │   │   └── expected_result.json
│   │   └── ...
│   └── ...
├── collected_results/       # Actual test results with timestamps
│   ├── bert-base-uncased/
│   │   ├── cpu/
│   │   │   └── 20250310_120000/
│   │   └── ...
│   └── summary/             # Summary reports from test runs
├── model_documentation/     # Generated documentation
│   ├── bert-base-uncased/
│   │   ├── cpu_implementation.md
│   │   └── ...
│   └── ...
└── runners/
    └── end_to_end/          # End-to-end testing framework scripts
        ├── integrated_component_test_runner.py   # Main script for integrated testing
        ├── template_validation.py                # Validation and comparison logic
        ├── model_documentation_generator.py      # Documentation generator
        └── simple_utils.py                       # Utility functions
```

## Template-Driven Approach

The framework uses a template-driven approach to generate components, focusing on fixing generators rather than individual files:

1. **Template Retrieval**: Gets templates from a central database or file system
2. **Parameterization**: Fills in model-specific and hardware-specific details
3. **Validation**: Checks that generated components meet requirements
4. **Combined Testing**: Tests all components together as a unified whole

This approach ensures:
- Consistent implementation patterns
- Easier maintenance (fix one template instead of hundreds of files)
- Improved cross-platform support
- Better documentation

## Testing Process

The integrated testing workflow follows these steps:

1. **Component Generation**:
   - Retrieves appropriate templates for model/hardware combination
   - Fills in parameters and generates skill, test, and benchmark files
   - Validates generated files for correctness

2. **Test Execution**:
   - Runs the test file with the skill implementation
   - Captures test results and success/failure status

3. **Benchmark Execution**:
   - Runs the benchmark file with the skill implementation
   - Captures performance metrics and statistics

4. **Result Storage**:
   - Saves all results to timestamped directories
   - Updates expected results if requested
   - Stores results in DuckDB database if available

5. **Result Comparison**:
   - Compares actual results with expected results
   - Uses configurable tolerances for numerical values
   - Provides detailed comparison information

6. **Documentation Generation**:
   - Creates comprehensive Markdown documentation
   - Includes implementation details, API documentation, and results
   - Organizes documentation by model and hardware platform

## Model Categories

The framework supports the following model families:

| Family | Description | Examples |
|--------|-------------|----------|
| text-embedding | Text embedding models | BERT, Sentence Transformers |
| text-generation | Text generation models | OPT, T5, Falcon |
| vision | Computer vision models | ViT, DETR, CLIP (vision) |
| audio | Audio processing models | Whisper, Wav2Vec2, CLAP |
| multimodal | Multimodal models | CLIP, LLaVA, FLAVA |

## Hardware Platforms

The framework supports the following hardware platforms:

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

The framework prioritizes testing on cpu, cuda, openvino, and webgpu platforms by default.

## Database Integration

The framework integrates with DuckDB for efficient storage and retrieval of test results:

1. **Result Storage**: Stores all test and benchmark results with rich metadata
2. **Version Tracking**: Includes git hash and timestamp information
3. **Efficient Queries**: Enables complex analysis of performance trends
4. **Simulation Detection**: Tracks whether hardware was simulated

```bash
# Use a specific database path
python integrated_component_test_runner.py --model bert-base-uncased --hardware cpu --db-path ./benchmark_db.duckdb

# Disable database storage
python integrated_component_test_runner.py --model bert-base-uncased --hardware cpu --no-db
```

## Documentation Generation

The framework can automatically generate detailed documentation for model implementations:

```bash
# Generate documentation for a specific model and hardware
python integrated_component_test_runner.py --model bert-base-uncased --hardware cpu --generate-docs
```

The generated documentation includes:
- Implementation details and architecture
- API documentation with method descriptions
- Usage examples
- Test results and performance metrics
- Hardware-specific optimizations
- Limitations and recommendations

## Expected Results and Regression Testing

The framework supports regression testing by comparing results with expected baselines:

```bash
# Update expected results
python integrated_component_test_runner.py --model bert-base-uncased --hardware cpu --update-expected

# Compare with expected results
python integrated_component_test_runner.py --model bert-base-uncased --hardware cpu
```

This ensures that changes to the implementation don't unexpectedly affect results.

## Best Practices

1. **Update Expected Results After Changes**:
   When making significant changes to model implementations or templates, update the expected results:
   ```bash
   python integrated_component_test_runner.py --model your-model --hardware your-hardware --update-expected
   ```

2. **Generate Documentation**:
   Keep documentation up-to-date when making changes:
   ```bash
   python integrated_component_test_runner.py --model your-model --hardware your-hardware --generate-docs
   ```

3. **Clean Up Old Results**:
   Periodically clean up old collected results to save disk space:
   ```bash
   python integrated_component_test_runner.py --clean-old-results --days 14
   ```

4. **Focus on Templates**:
   When fixing issues, focus on the template generators rather than individual files.
   
5. **Test Across Hardware Platforms**:
   Ensure models work on all necessary hardware platforms:
   ```bash
   python integrated_component_test_runner.py --model your-model --hardware cpu,cuda,openvino,webgpu
   ```

6. **Maintain Consistency**:
   Use the same templates and patterns across related models for maintainability.

## Troubleshooting

1. **Tests Are Failing But Implementation Looks Correct**:
   - Check if expected results need updating: `--update-expected`
   - Examine the differences in test results
   - Adjust the tolerance level if precision differences are expected: `--tolerance 0.05`

2. **Documentation Is Not Generating Correctly**:
   - Check that components are being generated correctly
   - Verify the model and hardware names are correct
   - Check template variable substitution

3. **Database Integration Issues**:
   - Verify the DuckDB installation with: `pip install duckdb==0.9.2`
   - Check database path permissions
   - Try using `--no-db` to disable database integration temporarily

4. **Template Issues**:
   - Check template rendering parameters
   - Verify template database path
   - Use `--verbose` to see detailed template operations

5. **Visualization Dashboard and Integrated Reports Issues**:
   - For comprehensive troubleshooting of the dashboard and reports system, refer to [TROUBLESHOOTING_GUIDE.md](./TROUBLESHOOTING_GUIDE.md)
   - For detailed understanding of the system architecture, refer to [SYSTEM_ARCHITECTURE.md](./SYSTEM_ARCHITECTURE.md)
   - These guides provide detailed information on:
     - Dashboard process management issues
     - Database connectivity problems
     - Report generation failures
     - Visualization rendering problems
     - Browser integration challenges
     - Combined workflow issues
     - CI/CD integration troubleshooting
     - Component interactions and data flow
     - Process management and error handling
     - System integration points and extensibility

## Contact

If you have questions or need support with the integrated component testing framework, please contact the infrastructure team.