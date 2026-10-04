# Generators Directory Structure

## Overview

The `generators` directory contains all code generation tools for the IPFS Accelerate project, organized into subdirectories by functionality.

## Directory Layout

```
scripts/generators/
├── benchmark_scripts/generators/     # Benchmark code generation
│   ├── benchmark_generator.py
│   ├── report_generator.py
│   └── create_benchmark_schema.py
├── creators/                 # Creator utilities
│   ├── create_generator.py
│   ├── create_minimal_generator.py
│   └── create_real_webgpu_implementation.py
├── fixes/                    # Fixes for generators
│   ├── fix_all_generators.py
│   ├── fix_template.py
│   └── fix_generators_phase16.py
├── models/                   # Model implementations
│   ├── skill_hf_bert.py
│   ├── skill_hf_llama.py
│   └── skill_hf_vit.py
├── runners/                  # Runner scripts
│   ├── run_fixed_test_generator.py
│   ├── run_hardware_tests.py
│   └── web/                  # Web-specific runners
├── skill_scripts/generators/         # Skill generation
│   ├── skill_generator.py
│   └── template_processor.py
├── template_scripts/generators/      # Template operations
│   ├── simple_template_validator.py
│   └── template_inheritance_system.py
├── templates/                # Templates
│   ├── add_template.py
│   └── model_templates/      # Model-specific templates
│       ├── template_bert.py
│       └── template_vit.py
├── test_scripts/generators/          # Test generation
│   ├── simple_test_generator.py
│   ├── merged_test_generator.py
│   └── hardware_test_generator.py
└── utils/                    # Utilities
    ├── validate_generator_improvements.py
    └── verify_generator_fixes.py
```

## Key Files

### Core Generators
- `test_scripts/generators/merged_test_generator.py`: Main test generator
- `test_scripts/generators/simple_test_generator.py`: Simplified test generator
- `skill_scripts/generators/skill_generator.py`: Skill generator
- `benchmark_scripts/generators/benchmark_generator.py`: Benchmark generator

### Template System
- `template_scripts/generators/template_validator.py`: Template validation
- `template_scripts/generators/template_inheritance_system.py`: Template inheritance
- `templates/model_templates/`: Model-specific templates

### Runner Scripts
- `runners/run_fixed_test_generator.py`: Run test generators
- `runners/run_hardware_tests.py`: Run hardware tests
- `runners/run_all_skill_tests.py`: Run skill tests

### Utilities
- `utils/validate_generator_improvements.py`: Validate improvements
- `utils/verify_generator_fixes.py`: Verify fixes
- `creators/create_generator.py`: Create generators
