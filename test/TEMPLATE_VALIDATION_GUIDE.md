# Template Validation System Guide

## Overview

The Template Validation System is a comprehensive framework for validating and verifying template files used in test generation. This document explains how to use the enhanced validation system with support for hardware platforms, template variables, and generator compatibility.

## Key Components

1. **Template Validator** (`template_validator.py`): Core validation system with multiple validation rules
2. **Simple Template Validator** (`simple_template_validator.py`): Lightweight validator for basic checks
3. **Template Database** (`hardware_test_templates/template_database.py`): DuckDB-based storage for templates
4. **Template System Check** (`run_template_system_check.py`): Integration testing for all template system components

## Validation Features

The template validator checks for:

1. **Syntax Validation**: Ensures templates have valid Python syntax
2. **Import Validation**: Checks for required imports like os, sys, torch, logging
3. **Class Structure**: Verifies proper test class naming and methods
4. **ResourcePool Usage**: Validates resource sharing and model caching
5. **Hardware Awareness**: Checks for proper device handling and allocation
6. **Cross-Platform Support**: Validates support for multiple hardware platforms
   - CPU, CUDA (required)
   - ROCm/AMD, MPS/Apple, OpenVINO/Intel (recommended)
   - WebNN, WebGPU (web platforms)
   - Qualcomm AI Engine (mobile/edge acceleration)
7. **Template Variables**: Validates required variables like model_name, model_class
8. **Generator Compatibility**: Ensures templates work with all generator types
   - merged_test_generator
   - integrated_skillset_generator
   - fixed_merged_test_generator
   - simple_test_generator

## Using the Template Validator

### Basic Usage

```bash
# Validate a single template file
python template_validator.py --file path/to/template.py

# Validate all templates in a directory
python template_validator.py --dir path/to/templates

# Validate all templates in the database
python template_validator.py --all-db --db-path ./template_db.duckdb

# Validate templates for a specific model family
python template_validator.py --family bert --db-path ./template_db.duckdb

# Validate templates for a specific modality
python template_validator.py --modality text --db-path ./template_db.duckdb
```

### Advanced Usage

```bash
# Validate templates with specific generator compatibility
python template_validator.py --all-db --generator-type fixed --db-path ./template_db.duckdb

# Validate templates with all generator types
python template_validator.py --all-db --validate-all-generators --db-path ./template_db.duckdb

# Generate a validation report
python template_validator.py --all-db --report validation_report.md --db-path ./template_db.duckdb

# Store validation results in the database
python template_validator.py --all-db --store-in-db --db-path ./template_db.duckdb

# Show validation history for a specific model
python template_validator.py --history bert --db-path ./template_db.duckdb

# Test hardware compatibility
python template_validator.py --test-compatibility --dir path/to/templates
```

## Hardware Platform Support

Templates should support multiple hardware platforms, and the validator checks for proper support:

```python
# Required platforms
"cpu"      # Basic CPU support
"cuda"     # NVIDIA GPU support

# Recommended platforms
"mps"      # Apple Silicon GPU support
"rocm"     # AMD GPU support
"openvino" # Intel hardware acceleration

# Web platforms
"webnn"    # Web Neural Network API
"webgpu"   # Web GPU API

# Mobile/edge acceleration
"qualcomm" # Qualcomm AI Engine/Hexagon DSP
```

## Generator Compatibility

The validator can check if templates are compatible with different generator types:

```bash
# Check compatibility with merged generator
python template_validator.py --file template.py --generator-type merged

# Check compatibility with integrated skillset generator
python template_validator.py --file template.py --generator-type integrated_skillset

# Check compatibility with fixed generator
python template_validator.py --file template.py --generator-type fixed

# Check compatibility with simple generator
python template_validator.py --file template.py --generator-type simple

# Check compatibility with all generators
python template_validator.py --file template.py --validate-all-generators
```

## Template Database Integration

The validator integrates with the DuckDB-based template database:

```bash
# Validate all templates in database
python template_validator.py --all-db --db-path ./template_db.duckdb

# Store validation results in database
python template_validator.py --all-db --store-in-db --db-path ./template_db.duckdb

# Get validation history from database
python template_validator.py --history bert --history-limit 5 --db-path ./template_db.duckdb
```

## Template Structure Requirements

The validator checks for proper template structure:

1. **Imports**: Required imports for functionality
   ```python
   import os
   import sys
   import torch
   import logging
   ```

2. **Class Structure**: Proper test class naming and methods
   ```python
   class TestModel:
       @classmethod
       def setup_class(cls):
           # Setup code
           pass
           
       def test_inference(self):
           # Test code
           pass
           
       @classmethod
       def teardown_class(cls):
           # Cleanup code
           pass
   ```

3. **Resource Pool**: Proper resource management
   ```python
   from resource_pool import get_global_resource_pool
   
   pool = get_global_resource_pool()
   model = pool.get_model(model_name)
   ```

4. **Hardware Awareness**: Proper device handling
   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model = model.to(device)
   ```

5. **Template Variables**: Required template variables
   ```python
   model_name = "{{ model_name }}"
   model_class = "{{ model_class }}"
   ```

## Generating Validation Reports

The validator can generate comprehensive validation reports:

```bash
# Generate a markdown report
python template_validator.py --all-db --report validation_report.md

# Generate a JSON report
python template_validator.py --all-db --output validation_results.json

# Generate both and store in database
python template_validator.py --all-db --report validation_report.md --output validation_results.json --store-in-db
```

Report includes:
- Summary of valid/invalid templates
- Hardware platform support statistics
- Details about invalid templates and their errors
- Generator compatibility information

## Template System Verification

You can verify the entire template system using the integration check script:

```bash
python run_template_system_check.py
```

This script performs end-to-end validation of:
1. Template database creation
2. Template validation
3. Test generation
4. Hardware detection
5. Vision model templates
6. Qualcomm support
7. Generator compatibility
8. Test execution

## Advanced Configuration

You can enable verbose logging for more detailed output:

```bash
python template_validator.py --file template.py --verbose
```

## Best Practices

1. **Complete Hardware Support**: Include support for all hardware platforms
2. **Generator Compatibility**: Ensure templates work with all generator types
3. **Template Variables**: Include all required template variables
4. **Resource Management**: Use ResourcePool for efficient resource sharing
5. **Regular Validation**: Run validation as part of CI/CD pipeline
6. **History Tracking**: Store validation results in the database for tracking

## Troubleshooting

### Common Issues

1. **Missing Hardware Support**: Add support for missing hardware platforms
2. **Generator Incompatibility**: Ensure templates include imports for all generators
3. **Invalid Template Variables**: Check for missing or invalid template variables
4. **Resource Pool Issues**: Add proper ResourcePool usage

### Using Verbose Mode

For detailed debugging information, use verbose mode:

```bash
python template_validator.py --file template.py --verbose
```

## Next Steps

1. **Automated Fixes**: Create tools to automatically fix template issues
2. **CI/CD Integration**: Add template validation to CI/CD pipeline
3. **Extended Hardware Support**: Add support for more hardware platforms
4. **Generator Compatibility**: Improve generator compatibility checks
5. **Template Migration**: Develop tools for template migration between versions