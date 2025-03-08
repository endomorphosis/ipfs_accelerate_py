# IPFS Accelerate Generators

This directory contains generators for creating tests, skills, and benchmarks for the IPFS Accelerate Python framework. The generators are organized into three main categories:

## Directory Structure

```
generators/
├── test_generators/         # Generators for test files
│   ├── model_test_generator.py        # Creates tests for specific model types
│   ├── merged_test_generator.py       # Advanced test generator with merged features
│   ├── hardware_test_generator.py     # Hardware-specific test generator
│   └── generate_key_tests.py          # Generates tests for key model types
├── skill_generators/        # Generators for skills
│   ├── skill_generator.py             # Creates skill implementations
│   ├── template_processor.py          # Processes templates for skill generation
│   ├── template_validator.py          # Validates template configurations
│   └── create_template_db.py          # Creates template databases
└── benchmark_generators/    # Generators for benchmarks
    ├── benchmark_generator.py         # Creates benchmark programs
    └── report_generator.py            # Generates benchmark reports
```

## Usage

### Test Generators

Test generators create test files for running models on different hardware platforms:

```python
# Using model_test_generator.py (formerly simple_test_generator.py)
python -m generators.test_generators.model_test_generator -g bert -p cpu,cuda,openvino -o test_bert.py

# Using merged_test_generator.py (formerly fixed_merged_test_generator.py)
python -m generators.test_generators.merged_test_generator --generate bert --platform webgpu,webnn

# Using hardware_test_generator.py (formerly qualified_test_generator.py)
python -m generators.test_generators.hardware_test_generator -g bert-base-uncased -p cpu,cuda,openvino,qualcomm
```

### Skill Generators

Skill generators create skill implementations for different model types:

```python
# Using skill_generator.py (formerly integrated_skillset_generator.py)
python -m generators.skill_generators.skill_generator --model bert --cross-platform

# Using template_processor.py (formerly fixed_template_generator.py)
python -m generators.skill_generators.template_processor --create-template bert

# Create a template database
python -m generators.skill_generators.create_template_db
```

### Benchmark Generators

Benchmark generators create benchmark programs and reports:

```python
# Using benchmark_generator.py (formerly benchmark_timing_report.py)
python -m generators.benchmark_generators.benchmark_generator --models bert,t5,vit --output benchmark_report.html

# Using report_generator.py (formerly benchmark_visualizer.py)
python -m generators.benchmark_generators.report_generator --input benchmark_results.json --output benchmark_report.html
```

## Integration with Other Components

The generators integrate with the following components:

1. **DuckDB Template Database**: Generators use template databases for storing and retrieving templates.
2. **Worker Architecture**: Generators leverage the worker architecture for hardware detection and model information.
3. **Benchmark Database**: Benchmark generators integrate with the DuckDB benchmark database.

## Configuration

Common configuration options are available in the `config.py` file:

```python
from generators.config import GeneratorConfig

# Get default hardware backends
hardware_backends = GeneratorConfig.get_default_hardware_backends()

# Get key model types
model_types = GeneratorConfig.get_key_model_types()
```

## Utilities

Common utilities are available in the `utils.py` file:

```python
from generators.utils import setup_logger, ensure_directory, template_variable_substitution

# Set up logger
logger = setup_logger("my_generator")

# Ensure directory exists
ensure_directory("output_dir")

# Substitute variables in template
template = "Model type: {model_type}, Hardware: {hardware}"
variables = {"model_type": "bert", "hardware": "cuda"}
result = template_variable_substitution(template, variables)
```

## Relationship to IPFS Accelerate SDK Enhancement

These generators will be used as part of the Python SDK Enhancement planned for 2025. The generators will:

1. Create test files for different model types and hardware platforms
2. Generate skill implementations for the worker architecture
3. Create benchmark programs for performance testing
4. Generate reports for analyzing benchmark results

The generators are designed to work with both the current architecture and the planned SDK enhancements.