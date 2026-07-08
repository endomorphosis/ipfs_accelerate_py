# TypeScript SDK Migration Tools Guide

This document provides details on using the tools developed for the WebGPU/WebNN migration to TypeScript, including options, common usage patterns, and troubleshooting tips.

## Key Tools Overview

1. **`validate_import_paths.py`**: Validates and fixes import path issues in TypeScript files
2. **`setup_typescript_test.py`**: Sets up TypeScript compilation environment and fixes type issues  
3. **Python-to-TypeScript converter**: Converts Python code to TypeScript

## 1. Import Path Validation Tool

### Purpose
The `validate_import_paths.py` script validates and fixes import paths in TypeScript files. It checks if imported modules exist, fixes invalid imports, and helps eliminate circular dependencies.

### Usage

Basic validation (no fixes):
```bash
python validate_import_paths.py --target-dir ../ipfs_accelerate_js
```

Validation with automatic fixes:
```bash
python validate_import_paths.py --target-dir ../ipfs_accelerate_js --fix
```

With verbose output:
```bash
python validate_import_paths.py --target-dir ../ipfs_accelerate_js --fix --verbose
```

### Key Features

- Creates index.ts files in key directories
- Fixes common import patterns
- Detects circular dependencies
- Handles various import syntax formats
- Generates validation reports

### Output

The tool produces:
- Log file: `validate_import_paths.log`
- Report: `import_validation_report.md`

## 2. TypeScript Setup Tool

### Purpose
The `setup_typescript_test.py` script configures the TypeScript environment, installs dependencies, fixes common type issues, and runs the TypeScript compiler for validation.

### Usage

Basic setup (no installation or compilation):
```bash
python setup_typescript_test.py --target-dir ../ipfs_accelerate_js
```

Install dependencies:
```bash
python setup_typescript_test.py --target-dir ../ipfs_accelerate_js --install
```

Fix common type issues:
```bash
python setup_typescript_test.py --target-dir ../ipfs_accelerate_js --fix-types
```

Run TypeScript compiler:
```bash
python setup_typescript_test.py --target-dir ../ipfs_accelerate_js --compile
```

Complete setup (all options):
```bash
python setup_typescript_test.py --target-dir ../ipfs_accelerate_js --install --fix-types --compile
```

### Key Features

- Creates/updates tsconfig.json
- Ensures package.json has TypeScript dependencies
- Creates basic type definitions for WebGPU and WebNN
- Fixes common type issues in TypeScript files
- Runs TypeScript compiler for validation

### Output

The tool produces:
- Log file: `typescript_test.log`
- Error log: `typescript_errors.log`
- Error summary: `typescript_error_summary.md`

## 3. Advanced Type Fixing

To fix more complex TypeScript errors, we need to address:

1. **Python-specific syntax**:
   - List/tuple destructuring → Array destructuring
   - Dictionary syntax → Object literal syntax  
   - Class method definitions with self → this

2. **Type annotations**:
   - Function parameters and return types
   - Class property types
   - Generic types

3. **Fix common patterns**:
   - Replace Python docstrings with JSDoc comments
   - Fix Python imports to TypeScript imports
   - Convert Python exceptions to try/catch

## Troubleshooting

### Import Paths

If you encounter persistent import path issues:
```bash
# Run multiple passes
python validate_import_paths.py --target-dir ../ipfs_accelerate_js --fix
python validate_import_paths.py --target-dir ../ipfs_accelerate_js --fix
```

### TypeScript Errors

For files with severe syntax issues:
1. Check the original Python file
2. Manually convert critical sections
3. Create simplified implementations for complex components

### Dependency Issues

If npm install fails:
```bash
# The setup script now uses --legacy-peer-deps flag by default
python setup_typescript_test.py --target-dir ../ipfs_accelerate_js --install
```

## Best Practices

1. **Staged Approach**:
   - Fix core/infrastructure files first
   - Then fix model implementations
   - Focus on getting a subset working initially

2. **Prioritize Files**:
   - Fix files with fewer errors first
   - Focus on files needed for core functionality
   - Leave test files for later

3. **Testing Strategy**:
   - Validate imports first
   - Then fix type issues
   - Finally test actual functionality

## Next Steps

After completing the migration:

1. Set up bundling with Rollup or Webpack
2. Create comprehensive documentation
3. Set up automated tests
4. Configure CI/CD pipeline
5. Publish to NPM

## Additional Resources

- [TypeScript Documentation](https://www.typescriptlang.org/docs/)
- [WebGPU Specification](https://gpuweb.github.io/gpuweb/)
- [WebNN API](https://webmachinelearning.github.io/webnn/)
- [TypeDoc Documentation](https://typedoc.org/)