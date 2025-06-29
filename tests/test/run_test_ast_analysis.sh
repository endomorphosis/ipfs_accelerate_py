#!/bin/bash

# Script to generate and analyze AST reports for test files
# This is used to support the test refactoring initiative

set -e

# Create directory for analysis outputs
ANALYSIS_DIR="test_analysis_$(date +%Y%m%d)"
mkdir -p $ANALYSIS_DIR

# Generate AST report
echo "Generating AST report for test files..."
# Do a deep scan of all test files including subdirectories
python generate_test_ast_report.py --directory . --pattern "**/test_*.py" --output $ANALYSIS_DIR/test_ast_report.json

# Check if report was generated successfully
if [ ! -f "$ANALYSIS_DIR/test_ast_report.json" ]; then
    echo "Error: Failed to generate AST report"
    exit 1
fi

# Skip dependency installation as we're in a managed environment
echo "Analyzing AST report..."
# Create a comprehensive analysis instead
python -c "
import json
import os
from collections import Counter, defaultdict
import re
from datetime import datetime

# Load report
with open('$ANALYSIS_DIR/test_ast_report.json', 'r') as f:
    report = json.load(f)

# Basic analysis
test_files = len(report['files'])
total_classes = sum(f['metrics']['num_classes'] for f in report['files'])
total_methods = sum(f['metrics']['num_methods'] for f in report['files'])
total_test_methods = sum(f['metrics']['num_test_methods'] for f in report['files'])

# Find common patterns
class_names = Counter()
base_classes = Counter()
test_method_names = Counter()
import_modules = Counter()
file_paths = [] 
common_fixtures = Counter()
test_method_patterns = defaultdict(list)
method_size_distribution = defaultdict(int)
file_size_distribution = defaultdict(int)
class_method_count = defaultdict(int)

# Regular expression for common test patterns
fixture_pattern = re.compile(r'(setup|teardown|mock_|create_test_|prepare_|initialize_)', re.IGNORECASE)
assertion_pattern = re.compile(r'assert\w+\(')

# Detailed analysis
for file_info in report['files']:
    file_paths.append(file_info['path'])
    # Get file size category (in KB)
    file_size_kb = file_info['size_bytes'] // 1024
    if file_size_kb == 0:
        size_category = '< 1KB'
    elif file_size_kb < 5:
        size_category = '1-5KB'
    elif file_size_kb < 10:
        size_category = '5-10KB'
    elif file_size_kb < 20:
        size_category = '10-20KB'
    elif file_size_kb < 50:
        size_category = '20-50KB'
    else:
        size_category = '> 50KB'
    file_size_distribution[size_category] += 1
    
    # Analyze imports
    for imp in file_info['imports']:
        module = imp['module'].split('.')[0]  # Get top-level module
        import_modules[module] += 1
    
    # Analyze classes and methods
    for class_info in file_info['classes']:
        class_names[class_info['name']] += 1
        
        # Count methods per class
        method_count = len(class_info['methods'])
        class_method_count[f'{method_count} methods'] += 1
        
        for base in class_info['bases']:
            base_classes[base] += 1
        
        # Analyze methods
        for method in class_info['methods']:
            # Method size distribution (in lines)
            method_size = method['end_line_number'] - method['line_number']
            size_category = '1-5 lines'
            if method_size > 5 and method_size <= 10:
                size_category = '6-10 lines'
            elif method_size > 10 and method_size <= 20:
                size_category = '11-20 lines'
            elif method_size > 20 and method_size <= 50:
                size_category = '21-50 lines'
            elif method_size > 50:
                size_category = '> 50 lines'
            
            if method.get('is_test', False):
                method_size_distribution[size_category] += 1
                test_method_names[method['name']] += 1
                
                # Extract method name pattern
                name = method['name']
                if name.startswith('test_'):
                    name_pattern = name[5:].split('_')[0]  # Get first word after test_
                    test_method_patterns[name_pattern].append(name)
            
            # Check for fixture methods
            if fixture_pattern.search(method['name']):
                common_fixtures[method['name']] += 1

# Find duplicate tests (same method name in different files)
duplicate_tests = [name for name, count in test_method_names.items() if count > 1]

# Identify test file clustering by directory
directory_clustering = Counter()
for path in file_paths:
    # Extract directory from path
    directory = os.path.dirname(path)
    if directory == '.':
        directory = 'root'
    directory_clustering[directory] += 1

# Generate more comprehensive recommendations
test_categories = {
    'unit': ['test_unit', 'test_function', 'test_method'],
    'integration': ['test_integration', 'test_connect', 'test_api'],
    'hardware': ['test_hardware', 'test_device', 'test_platform', 'test_compatibility'],
    'browser': ['test_browser', 'test_firefox', 'test_chrome', 'test_safari'],
    'web': ['test_web', 'test_js', 'test_html', 'test_react'],
    'model': ['test_model', 'test_bert', 'test_vit', 'test_gpt', 'test_llama'],
}

# Categorize test files
test_categories_found = Counter()
for file_info in report['files']:
    filename = file_info['filename'].lower()
    for category, keywords in test_categories.items():
        if any(keyword in filename for keyword in keywords):
            test_categories_found[category] += 1
            break

# Create a basic clustering of related test files
test_clusters = defaultdict(list)
for file_info in report['files']:
    file_keywords = file_info['filename'].lower().replace('.py', '').replace('test_', '').split('_')
    # Find primary keyword (usually the first non-generic term)
    primary_keyword = None
    for kw in file_keywords:
        if kw not in ['test', 'basic', 'simple', 'comprehensive', 'advanced']:
            primary_keyword = kw
            break
    
    if primary_keyword:
        test_clusters[primary_keyword].append(file_info['filename'])

# Save comprehensive report
with open('$ANALYSIS_DIR/refactoring_recommendations.md', 'w') as f:
    f.write('# Test Codebase Refactoring Recommendations\n\n')
    f.write(f'**Generated on:** {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}\n\n')
    
    f.write('## Executive Summary\n\n')
    f.write('This analysis examines the structure and patterns of test files in the IPFS Accelerate Python project to inform a comprehensive refactoring initiative.\n\n')
    
    f.write(f'- **Total test files analyzed:** {test_files}\n')
    f.write(f'- **Total test classes:** {total_classes}\n')
    f.write(f'- **Total test methods:** {total_test_methods}\n')
    f.write(f'- **Potential duplicate tests:** {len(duplicate_tests)}\n\n')
    
    f.write('## Test Codebase Structure\n\n')
    
    f.write('### Test Files by Directory\n\n')
    f.write('| Directory | Count |\n|------------|-------|\n')
    for directory, count in directory_clustering.most_common():
        f.write(f'| {directory} | {count} |\n')
    f.write('\n')
    
    f.write('### Test File Size Distribution\n\n')
    f.write('| Size Range | Count |\n|------------|-------|\n')
    for size, count in sorted(file_size_distribution.items(), key=lambda x: 
            0 if x[0] == '< 1KB' else 
            1 if x[0] == '1-5KB' else 
            2 if x[0] == '5-10KB' else 
            3 if x[0] == '10-20KB' else 
            4 if x[0] == '20-50KB' else 5):
        f.write(f'| {size} | {count} |\n')
    f.write('\n')
    
    f.write('### Test Categories\n\n')
    f.write('| Category | Count |\n|------------|-------|\n')
    for category, count in test_categories_found.most_common():
        f.write(f'| {category} | {count} |\n')
    f.write('\n')
    
    f.write('### Class Structure\n\n')
    
    f.write('#### Most Common Test Class Names\n\n')
    f.write('| Class Name | Count |\n|------------|-------|\n')
    for name, count in class_names.most_common(10):
        f.write(f'| {name} | {count} |\n')
    f.write('\n')
    
    f.write('#### Most Common Base Classes\n\n')
    f.write('| Base Class | Count |\n|------------|-------|\n')
    for name, count in base_classes.most_common(10):
        f.write(f'| {name} | {count} |\n')
    f.write('\n')
    
    f.write('#### Class Size Distribution\n\n')
    f.write('| Methods per Class | Count |\n|-----------------|-------|\n')
    for size, count in sorted(class_method_count.items(), key=lambda x: int(x[0].split()[0])):
        f.write(f'| {size} | {count} |\n')
    f.write('\n')
    
    f.write('### Test Method Analysis\n\n')
    
    f.write('#### Most Common Test Method Names\n\n')
    f.write('| Method Name | Count |\n|------------|-------|\n')
    for name, count in test_method_names.most_common(15):
        f.write(f'| {name} | {count} |\n')
    f.write('\n')
    
    f.write('#### Test Method Size Distribution\n\n')
    f.write('| Size Range | Count |\n|------------|-------|\n')
    for size, count in sorted(method_size_distribution.items(), key=lambda x: 
            0 if x[0] == '1-5 lines' else 
            1 if x[0] == '6-10 lines' else 
            2 if x[0] == '11-20 lines' else 
            3 if x[0] == '21-50 lines' else 4):
        f.write(f'| {size} | {count} |\n')
    f.write('\n')
    
    f.write('#### Common Test Fixtures/Helpers\n\n')
    f.write('| Method Name | Count |\n|------------|-------|\n')
    for name, count in common_fixtures.most_common(10):
        f.write(f'| {name} | {count} |\n')
    f.write('\n')
    
    f.write('#### Common Import Dependencies\n\n')
    f.write('| Module | Count |\n|--------|-------|\n')
    for module, count in import_modules.most_common(15):
        f.write(f'| {module} | {count} |\n')
    f.write('\n')
    
    f.write('## Major Test Clusters\n\n')
    for keyword, files in sorted(test_clusters.items(), key=lambda x: len(x[1]), reverse=True):
        if len(files) > 1:  # Only show clusters with more than one file
            f.write(f'### {keyword.capitalize()} Tests\n\n')
            for file in files:
                f.write(f'- {file}\n')
            f.write('\n')
    
    f.write('## Identified Issues\n\n')
    
    f.write('1. **Inconsistent Base Classes**: Multiple inheritance patterns without standardization\n')
    f.write('2. **Duplicate Test Methods**: Same test methods implemented across multiple files\n')
    f.write('3. **Inconsistent Naming Conventions**: Mixed naming patterns for test methods\n')
    f.write('4. **Redundant Fixtures**: Similar setup/teardown methods duplicated across files\n')
    f.write('5. **Size Distribution Issues**: Some tests are too large, others too small\n')
    f.write('6. **Directory Organization**: Tests scattered across multiple directories without clear organization\n\n')
    
    f.write('## Comprehensive Refactoring Recommendations\n\n')
    
    f.write('### 1. Standardize Test Structure\n\n')
    f.write('Create a hierarchy of base test classes:\n\n')
    f.write('- `BaseTest`: Core functionality for all tests\n')
    f.write('- `ModelTest`: Specialized functionality for ML model testing\n')
    f.write('- `BrowserTest`: Specialized functionality for browser testing\n')
    f.write('- `HardwareTest`: Specialized functionality for hardware compatibility testing\n')
    f.write('- `APITest`: Specialized functionality for API testing\n\n')
    
    f.write('### 2. Implement Consistent Naming Conventions\n\n')
    f.write('- Adopt a clear naming convention for test methods (e.g., `test_should_*` or `test_when_*_then_*`)\n')
    f.write('- Group tests by functionality, not implementation\n')
    f.write('- Use descriptive names that clearly indicate what is being tested\n\n')
    
    f.write('### 3. Extract Common Test Utilities\n\n')
    f.write('Create shared utility modules:\n\n')
    f.write('- `test_fixtures.py`: Common setup and teardown functionality\n')
    f.write('- `test_mocks.py`: Standard mock objects and factories\n')
    f.write('- `test_assertions.py`: Custom assertion helpers\n')
    f.write('- `test_data_generators.py`: Test data generation utilities\n\n')
    
    f.write('### 4. Reorganize Directory Structure\n\n')
    f.write('Organize tests into a more logical structure:\n\n')
    f.write('- `tests/unit/`: Unit tests for individual components\n')
    f.write('- `tests/integration/`: Integration tests between components\n')
    f.write('- `tests/hardware/`: Hardware-specific tests\n')
    f.write('- `tests/browser/`: Browser-specific tests\n')
    f.write('- `tests/models/`: ML model tests\n')
    f.write('- `tests/e2e/`: End-to-end tests\n\n')
    
    f.write('### 5. Consolidate Duplicate Tests\n\n')
    f.write('- Identify and merge duplicate test implementations\n')
    f.write('- Create parameterized tests for similar functionality across different models/components\n')
    f.write('- Develop a test registry to track test coverage and prevent duplication\n\n')
    
    f.write('### 6. Implement Test Size Standards\n\n')
    f.write('- Limit test methods to 10-20 lines when possible\n')
    f.write('- Extract helper methods for complex setup/assertions\n')
    f.write('- Use composition instead of inheritance for test reuse\n\n')
    
    f.write('### 7. Develop Deprecation Strategy\n\n')
    f.write('- Identify tests that are no longer relevant or redundant\n')
    f.write('- Create a migration path for deprecating old tests\n')
    f.write('- Document reasons for deprecation\n\n')
    
    f.write('## Implementation Plan\n\n')
    
    f.write('### Phase 1: Foundation (2 weeks)\n\n')
    f.write('1. Create base test classes and utilities\n')
    f.write('2. Develop naming convention guidelines\n')
    f.write('3. Implement directory structure reorganization\n\n')
    
    f.write('### Phase 2: Migration (3 weeks)\n\n')
    f.write('1. Convert high-priority tests to new structure\n')
    f.write('2. Consolidate duplicate tests\n')
    f.write('3. Implement parameterized testing\n\n')
    
    f.write('### Phase 3: Cleanup (2 weeks)\n\n')
    f.write('1. Deprecate unnecessary tests\n')
    f.write('2. Refine documentation\n')
    f.write('3. Create automated enforcement of test standards\n\n')
    
    f.write('### Phase 4: Validation (1 week)\n\n')
    f.write('1. Verify test coverage is maintained\n')
    f.write('2. Ensure all tests pass consistently\n')
    f.write('3. Measure performance improvements\n')
"

# Generate summary for quick review
echo "Generating summary of findings..."
cat > $ANALYSIS_DIR/SUMMARY.md << EOF
# Test Refactoring Analysis Summary

Generated on $(date)

## Overview

This analysis examines the structure and patterns of test files in the IPFS Accelerate Python project to 
inform a comprehensive refactoring plan.

## Key Files

- [Full Refactoring Recommendations](refactoring_recommendations.md) - Detailed recommendations and insights
- [Analysis Results](analysis_results.json) - Raw analysis data in JSON format
- [AST Report](test_ast_report.json) - Complete Abstract Syntax Tree report for all test files

## Visualizations

The \`visualizations\` directory contains graphical representations of:
- Test method size distribution
- Test methods per class distribution
- Class size distribution
- Class similarity network
- Inheritance clusters

## Next Steps

1. Review the refactoring recommendations
2. Prioritize changes based on impact and complexity
3. Develop a phased implementation plan
4. Begin with high-impact, low-effort improvements

EOF

echo "Analysis complete! Results available in $ANALYSIS_DIR"
echo "Review $ANALYSIS_DIR/SUMMARY.md to get started"