# Template Architecture Guide

## Architecture Overview

The template integration system uses a modular architecture to separate concerns and provide a flexible, maintainable framework for standardizing test files. This document explains the system's architecture and how the components interact.

## Component Architecture

### Core Components

1. **Template System**
   - `TEMPLATES_DIR`: Contains architecture-specific templates
   - `ARCHITECTURE_TYPES`: Maps architecture types to templates and registry names
   - Template files (`encoder_only_template.py`, `vision_text_template.py`, etc.)

2. **Model Configuration**
   - `MODEL_CONFIG`: Defines model-specific customizations
   - Model metadata (ID, class, task, processor, etc.)
   - Special handling code for specific model requirements

3. **Template Processing**
   - `customize_template()`: Applies model-specific customizations to templates
   - Template variable replacement
   - Registry entry addition
   - Special handling code insertion

4. **Integration Workflow**
   - Analysis of existing test files
   - Generation of new test files
   - Verification of generated files
   - Reporting and summary generation

5. **Utilities**
   - Syntax verification
   - Architecture type detection
   - Registry updates
   - File backup and restore

## Directory Structure

```
template_integration/
├── model_template_fixes.py        # Core template customization logic
├── template_integration_workflow.py # Integration workflow orchestration
├── fix_template_issues.py         # Targeted fixes for problematic models
├── apply_changes.py               # Applies changes to main codebase
├── README.md                      # System overview
├── TEMPLATE_INTEGRATION_GUIDE.md  # Comprehensive integration guide
├── ARCHITECTURE_GUIDE.md          # This architecture document
├── COMMAND_REFERENCE.md           # Command reference
├── manual_models_analysis.md      # Analysis report
└── template_integration_summary.md # Integration summary
```

## Data Flow

1. **Template Selection**
   - Architecture type determines template selection
   - Templates are read from `TEMPLATES_DIR`

2. **Model Configuration**
   - Model-specific configurations are retrieved from `MODEL_CONFIG`
   - Custom imports, special handling, and test inputs are extracted

3. **Template Customization**
   - Basic replacements are applied
   - Registry entries are added
   - Custom imports are inserted
   - Special handling code is inserted with proper indentation
   - Test inputs and processor classes are updated

4. **File Generation**
   - Customized template content is written to a new file
   - File is placed in the `FIXED_TESTS_DIR` directory
   - Original file is backed up if it exists

5. **Verification**
   - Generated file is checked for syntax errors
   - Indentation issues are identified
   - Success or failure is reported

6. **Integration**
   - Architecture type registries are updated
   - Generated files are applied to the main codebase

## Control Flow

### Template Integration Workflow

```
template_integration_workflow.py
  ├── analyze_manual_models()
  │    └── generate_analysis_report()
  ├── regenerate_model_tests()
  │    └── model_template_fixes.py::regenerate_all_models()
  ├── verify_regenerated_tests()
  │    └── generate_verification_report()
  └── generate_integration_summary()
```

### Model Template Fixes

```
model_template_fixes.py
  ├── get_template_path()
  ├── get_registry_name()
  ├── read_template()
  ├── customize_template()
  │    ├── Basic replacements
  │    ├── Handle registry entry
  │    ├── Handle custom imports
  │    ├── Handle special handling code
  │    ├── Update test inputs
  │    └── Update processor class
  ├── generate_test_file()
  ├── verify_test_file()
  ├── update_architecture_types()
  └── regenerate_all_models()
```

## Customization Points

1. **Adding New Model Types**
   - Add to `ARCHITECTURE_TYPES` in `model_template_fixes.py`
   - Create corresponding template in `TEMPLATES_DIR`

2. **Adding New Models**
   - Add entry to `MODEL_CONFIG` in `model_template_fixes.py`
   - Define architecture, model ID, class name, etc.
   - Add special handling code if needed

3. **Customizing Templates**
   - Modify template files in `TEMPLATES_DIR`
   - Update template variables

4. **Extending Special Handling**
   - Add model-specific special handling in `customize_template()`
   - Ensure proper indentation handling

## Implementation Notes

### Indentation Management

Proper indentation management is critical for valid Python syntax, especially for:

1. Special handling code for vision models (image creation)
2. Special handling code for speech models (audio creation)
3. Nested conditional statements and try/except blocks

Solution:
- Detect indentation levels from surrounding code
- Use f-strings with appropriate indentation prefixes
- Handle nested blocks with proper incremental indentation

### Registry Integration

Model registries are used to:
- Maintain a central repository of available models
- Define model metadata (architecture, class, task, etc.)
- Support dynamic model selection at runtime

The template system automatically adds new models to their respective architecture registries.

### Error Handling

The system includes robust error handling:
- Syntax error detection with line and context information
- Indentation issue identification
- Context tracking for better debugging
- Detailed error reporting

## Future Extensions

1. **Template Versioning**
   - Track template versions
   - Support migration between template versions

2. **Dependency Analysis**
   - Analyze model dependencies
   - Ensure all required libraries are imported

3. **Test Coverage Analysis**
   - Analyze test coverage of generated files
   - Identify gaps in testing

4. **Automated CI Integration**
   - Integrate with CI/CD pipelines
   - Automatically generate tests for new models