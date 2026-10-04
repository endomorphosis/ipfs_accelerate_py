# Template Verification and Regeneration

This directory contains tools for verifying template conformance of model test files and regenerating them to match the expected template structure.

## Problem Statement

Several manually created model test files do not follow the template structure used by the test generator system. This leads to:

1. **Code Inconsistency**: Different testing approaches for different models
2. **Limited Hardware Support**: Manual tests lack proper hardware detection
3. **Missing Mock Systems**: Manual tests don't properly handle CI/CD environments
4. **Syntax Errors**: Some manual tests have syntax issues
5. **Code Debt**: Changes to testing approach require updating each file individually

## Solution

This suite of tools provides a comprehensive solution for:

1. **Analyzing** the current state of template conformance
2. **Regenerating** model tests using the appropriate templates
3. **Verifying** that regenerated tests work correctly
4. **Updating** model registries and architecture mappings

## Tools Overview

### 1. `analyze_template_structure.py`

Analyzes template conformance of test files to identify issues.

Features:
- Identifies missing components in test files
- Compares file structure to reference templates
- Generates per-model recommendation reports
- Creates a summary report of all findings

Usage:
```bash
python analyze_template_structure.py [--verbose]
```

### 2. `regenerate_template_tests.py`

Regenerates test files using the appropriate templates.

Features:
- Maps models to architecture types
- Uses architecture-specific templates
- Updates model registry entries
- Ensures proper hardware detection and mock objects
- Verifies syntax of generated files

Usage:
```bash
# Regenerate a specific model
python regenerate_template_tests.py --model layoutlmv2 --verify

# Regenerate all models
python regenerate_template_tests.py --all --verify
```

### 3. `run_verification.py`

Runs a complete verification workflow to ensure template conformance.

Features:
- Runs analysis to identify issues
- Regenerates tests to fix issues
- Verifies regenerated tests for syntax and functionality
- Generates a comprehensive verification report

Usage:
```bash
python run_verification.py [--verbose]
```

## Model to Architecture Mapping

The following models are mapped to their correct architecture types:

| Model | Architecture Type | Template Used |
|-------|------------------|---------------|
| layoutlmv2 | vision-encoder-text-decoder | vision_text_template.py |
| layoutlmv3 | vision-encoder-text-decoder | vision_text_template.py |
| clvp | speech | speech_template.py |
| bigbird | encoder-decoder | encoder_decoder_template.py |
| seamless_m4t_v2 | speech | speech_template.py |
| xlm_prophetnet | encoder-decoder | encoder_decoder_template.py |

## Template Structure Components

Each template includes the following components:

1. **Hardware Detection**: Automatic detection of CPU, CUDA, MPS, OpenVINO
2. **Dependency Mocking**: Environment variable controls for CI/CD environments
3. **Model Registry**: Configuration of model-specific parameters
4. **Test Class**: Class-based testing framework with standardized methods
5. **Pipeline API**: Tests using the high-level pipeline API
6. **From_pretrained API**: Tests using the lower-level from_pretrained API
7. **Mock Objects**: Mock implementations for testing without dependencies
8. **Result Collection**: Standardized format for collecting test results
9. **Runtime Checking**: Runtime capability detection for hardware optimization
10. **Main Function**: CLI interface for running tests

## Workflow

### Recommended Process

1. **Analysis**: Run template analysis to understand the issues
   ```bash
   python analyze_template_structure.py
   ```

2. **Review**: Examine the analysis reports in the output directory

3. **Regeneration**: Regenerate tests with proper template structure
   ```bash
   python regenerate_template_tests.py --all --verify
   ```

4. **Verification**: Verify the regenerated tests
   ```bash
   python run_verification.py
   ```

5. **Integration**: Apply the changes to the main codebase after verification

## Output

Each tool generates output files in the `template_verification` directory:

- **Analysis Reports**: `*_recommendations.md` files with per-model recommendations
- **Summary Report**: `template_analysis_summary.md` with overall analysis
- **Verification Report**: `verification_report.md` with comprehensive results

## Benefits of Template Conformance

Ensuring all test files follow the template structure provides:

1. **Consistency**: All tests work the same way across models
2. **Maintainability**: Changes to testing approach only require template updates
3. **Hardware Support**: All tests work consistently across different hardware
4. **CI/CD Compatibility**: Tests work properly in continuous integration
5. **Result Standardization**: All tests report results in the same format
6. **Error Handling**: Consistent error handling across all tests