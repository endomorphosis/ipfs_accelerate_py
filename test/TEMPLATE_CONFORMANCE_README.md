# Template Conformance for Manually Created Tests

This document provides instructions for ensuring all test files in the project, especially manually created ones, conform to the established template structure.

## Issue Summary

Several manually created model test files do not follow the template structure used by the test generator system. This causes:

1. **Inconsistent Testing**: Manual tests use different validation logic
2. **Limited Hardware Support**: Manual tests lack proper hardware detection
3. **Missing Mock Systems**: Manual tests don't implement CI/CD mock objects
4. **Syntax Errors**: Some manual tests have syntax issues that prevent execution
5. **Code Debt**: Future changes require updating each file manually

## Affected Models

The manually created models that need to be regenerated are:

1. `layoutlmv2` (vision-text model)
2. `layoutlmv3` (vision-text model)
3. `clvp` (speech model)
4. `hf_bigbird` (encoder-decoder model)
5. `seamless_m4t_v2` (speech model)
6. `xlm_prophetnet` (encoder-decoder model)

## Architecture Mapping

Each model has been mapped to its correct architecture type:

| Model | Architecture Type | Template Used |
|-------|------------------|---------------|
| layoutlmv2 | vision-encoder-text-decoder | vision_text_template.py |
| layoutlmv3 | vision-encoder-text-decoder | vision_text_template.py |
| clvp | speech | speech_template.py |
| hf_bigbird | encoder-decoder | encoder_decoder_template.py |
| seamless_m4t_v2 | speech | speech_template.py |
| xlm_prophetnet | encoder-decoder | encoder_decoder_template.py |

## Solution: Fix Manual Models Script

A script called `fix_manual_models.py` has been created to automate the process of regenerating the manually created tests using the appropriate templates.

### Features

- Maps each model to its correct architecture type
- Uses appropriate template files from `skills/templates/`
- Generates proper test files with correct structure
- Updates architecture types in the test generator
- Verifies syntax of generated files
- Creates backups of existing files

### How to Use

1. **Verify Mode**: Check how the regeneration would work without applying changes

```bash
python fix_manual_models.py --verify
```

2. **Apply Mode**: Generate new tests and update architecture types

```bash
python fix_manual_models.py --verify --apply
```

3. **Run Generated Tests**: Test the newly generated files

```bash
cd skills/fixed_tests
python test_hf_layoutlmv2.py
python test_hf_clvp.py
# etc.
```

## Template Structure Benefits

The template system provides significant benefits:

1. **Hardware Detection**: Automatically detects and uses available hardware (CUDA, MPS, etc.)
2. **Mock Detection**: Creates mock objects when libraries are unavailable
3. **Unified Reporting**: Standardized result collection and formatting
4. **Error Handling**: Consistent error handling across all tests
5. **Environment Controls**: Environment variable support for CI/CD settings
6. **Future Compatibility**: Changes to templates propagate to all tests

## Implementation Status

All manually created models can now be regenerated using the template system, ensuring:

1. Proper syntax and indentation
2. Consistent hardware detection and fallbacks
3. Standardized mock objects for CI/CD environments
4. Unified result collection and reporting
5. Compatibility with future template changes

## Verification Process

Each regenerated test file is verified by:

1. Python syntax checking (using py_compile)
2. Architectural consistency validation
3. Hardware detection verification
4. Mock object functionality testing
5. Result format validation

## Future Improvements

Additional features to consider:

1. **Automation**: Integrate into CI/CD process to ensure all tests remain template-compliant
2. **Templates Update**: Enhance templates with more capabilities like batch processing
3. **Documentation**: Update model registry and architecture mapping documentation
4. **Parameter Configuration**: Add support for model-specific parameter configuration
5. **Comprehensive Testing**: Expand tests to cover more edge cases and hardware types