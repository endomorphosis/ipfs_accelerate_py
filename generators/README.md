# Generators Directory

This directory contains files for generating Python code for the IPFS Accelerate framework.

## Directory Structure

- **benchmark_generators/**: Generators for benchmark code
- **models/**: Model implementation files 
- **generated_tests/**: Generated test files that are ready to run
- **generated_skillsets/**: Generated skill implementation files
- **templates/**: Template files for test and skill generation
- **archive/**: Archived files and template generators that need template processing

## Note on Template Files and Syntax

Many template files intentionally have Python syntax errors when viewed directly because they contain template placeholders like `{{model_name}}` or `{model_name}`. These files should be used through the template instantiation system, not executed directly.

Template files with complex template syntax have been moved to archive/syntax_templates/.

## Phase 16: Test Generator Improvement

This directory contains update scripts to improve the Phase 16 test generators by addressing the duplicate hardware detection code and ensuring consistent hardware detection across all test generators and templates.

## Key Improvements

1. **Centralized Hardware Detection**
   - Eliminates duplicate hardware detection code across multiple files
   - Ensures consistent hardware detection in all test generators
   - Simplifies maintenance and future hardware platform additions
   - Improves memory usage by eliminating redundant imports

2. **Template System Integration**
   - Updates hardware template system to use centralized detection
   - Maintains backward compatibility for existing code
   - Ensures all templates use the same hardware detection logic
   - Improves hardware compatibility matrix consistency

3. **Web Platform Optimizations Standardization**
   - Centralizes web platform optimization flags and detection
   - Standardizes Firefox-specific WebGPU compute shader optimizations
   - Centralizes browser detection logic
   - Improves model sharding integration

## Usage

Run these scripts in sequence to update the test generation system:

```bash
# 1. Update main test generators
python update_generators/update_test_generators.py

# 2. Update hardware template system
python update_generators/update_hardware_template_system.py

# 3. Verify the changes
python test/centralized_hardware_detection/hardware_detection.py
```

After running these scripts, all test generators will use the centralized hardware detection system and produce more consistent tests with reliable cross-platform hardware support.

## Affected Files

### Main Generators
- `fixed_merged_test_generator.py`
- `merged_test_generator.py`
- `integrated_skillset_generator.py`
- `implementation_generator.py`
- `template_hardware_detection.py`

### Hardware Templates
- All files in `hardware_test_templates/` directory
- `template_database.json`

## Impact on Generated Tests

The generated tests will have:

1. More consistent hardware detection code
2. Better web platform optimizations
3. Improved compatibility with multi-platform testing
4. Better handling of Firefox-specific optimizations
5. More reliable test generation for audio and multimodal models