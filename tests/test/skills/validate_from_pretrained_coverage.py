#!/usr/bin/env python3

'''
Validate that all HuggingFace model classes with a from_pretrained method
have proper test coverage.

This script checks each test file to verify that the from_pretrained method
is correctly tested, and generates a report of any missing coverage.
'''

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TEST_DIR = './fixed_tests'
INVENTORY_FILE = 'model_class_inventory.json'


def load_inventory(inventory_file: str = INVENTORY_FILE) -> Dict[str, Any]:
    """Load the model inventory from a JSON file."""
    if not os.path.exists(inventory_file):
        logger.error(f"Inventory file not found: {inventory_file}")
        logger.error("Run generate_model_class_inventory.py first")
        sys.exit(1)
    
    try:
        with open(inventory_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading inventory: {e}")
        sys.exit(1)


def check_test_file(file_path: str) -> Tuple[bool, Dict[str, Any]]:
    """Check a test file for proper from_pretrained testing."""
    if not os.path.exists(file_path):
        return False, {}
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        results = {
            'has_from_pretrained_method': False,
            'calls_from_pretrained': False,
            'validates_output': False,
            'handles_errors': False,
            'models_tested': [],
        }
        
        # Check for method definition
        if re.search(r'def\s+test_from_pretrained', content):
            results['has_from_pretrained_method'] = True
        
        # Check for method call
        if '.from_pretrained(' in content:
            results['calls_from_pretrained'] = True
        
        # Check for output validation (checking results)
        validation_patterns = [
            r'assert\s+[^\n;]+\s*[!=]=\s*[^\n;]+',
            r'self\.assert',
            r'assertTrue',
            r'assertEqual',
            r'assertIsNotNone',
        ]
        if any(re.search(pattern, content) for pattern in validation_patterns):
            results['validates_output'] = True
        
        # Check for error handling
        error_patterns = [
            r'try\s*:',
            r'except\s+',
            r'assertRaises',
        ]
        if any(re.search(pattern, content) for pattern in error_patterns):
            results['handles_errors'] = True
        
        # Extract tested models
        model_id_patterns = [
            r'model_id\s*=\s*["\']([^"\']*)["\'](,|\s|\))',
            r'\.from_pretrained\(["\']([^"\']*)["\'](,|\s|\))',
        ]
        for pattern in model_id_patterns:
            for match in re.finditer(pattern, content):
                model_id = match.group(1)
                if model_id and model_id not in results['models_tested']:
                    results['models_tested'].append(model_id)
        
        # Overall assessment
        has_proper_test = results['has_from_pretrained_method'] and results['calls_from_pretrained']
        
        return has_proper_test, results
    except Exception as e:
        logger.warning(f"Error checking {file_path}: {e}")
        return False, {}


def analyze_test_coverage(inventory: Dict[str, Any], test_dir: str = TEST_DIR) -> Dict[str, Any]:
    """Analyze test coverage for all model classes."""
    # Initialize results
    results = {
        'generated_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_classes': inventory['total_classes'],
        'total_with_from_pretrained': inventory['coverage']['with_from_pretrained'],
        'total_tested': 0,
        'missing_tests': [],
        'partial_tests': [],
        'categories': {},
    }
    
    # Track classes by category
    for category, data in inventory['categories'].items():
        results['categories'][category] = {
            'total': data['count'],
            'with_from_pretrained': 0,
            'tested': 0,
            'missing': [],
            'partial': [],
        }
        
        for cls in data['classes']:
            # Skip if no from_pretrained method
            if not cls['has_from_pretrained']:
                continue
            
            results['categories'][category]['with_from_pretrained'] += 1
            
            # Skip if no test file
            test_file = cls['test_file']
            if not test_file:
                results['categories'][category]['missing'].append(cls['name'])
                results['missing_tests'].append((category, cls['name'], 'No test file'))
                continue
            
            # Check the test file
            has_proper_test, test_details = check_test_file(test_file)
            
            if has_proper_test:
                results['categories'][category]['tested'] += 1
                results['total_tested'] += 1
            else:
                # Add to missing or partial lists
                if test_details.get('has_from_pretrained_method') or test_details.get('calls_from_pretrained'):
                    # Partial test - has method or calls but doesn't fully test
                    results['categories'][category]['partial'].append(cls['name'])
                    results['partial_tests'].append((category, cls['name'], test_file, test_details))
                else:
                    # Missing test - no method or calls
                    results['categories'][category]['missing'].append(cls['name'])
                    results['missing_tests'].append((category, cls['name'], 'Test file exists but no from_pretrained test'))
    
    # Calculate percentages
    results['percent_tested'] = round(results['total_tested'] / results['total_with_from_pretrained'] * 100, 2) if results['total_with_from_pretrained'] > 0 else 0
    
    return results


def create_coverage_report(results: Dict[str, Any], output_file: str = 'from_pretrained_validation_report.md'):
    """Create a markdown report from the validation results."""
    with open(output_file, 'w') as f:
        f.write(f"# HuggingFace from_pretrained() Test Validation Report\n\n")
        f.write(f"Generated on: {results['generated_date']}\n\n")
        
        f.write(f"## Summary\n\n")
        f.write(f"- **Total model classes:** {results['total_classes']}\n")
        f.write(f"- **Classes with from_pretrained:** {results['total_with_from_pretrained']}\n")
        f.write(f"- **Classes with proper tests:** {results['total_tested']} ({results['percent_tested']}%)\n")
        f.write(f"- **Classes with partial tests:** {len(results['partial_tests'])}\n")
        f.write(f"- **Classes missing tests:** {len(results['missing_tests'])}\n\n")
        
        # Overall status
        if results['percent_tested'] == 100:
            f.write(f"✅ **All model classes with from_pretrained have proper tests!** ✅\n\n")
        else:
            f.write(f"⚠️ **Some model classes with from_pretrained are missing proper tests** ⚠️\n\n")
        
        f.write(f"## Coverage by Category\n\n")
        
        # Create table of category stats
        f.write(f"| Category | Total Classes | With from_pretrained | Tested | Coverage % |\n")
        f.write(f"|----------|---------------|----------------------|--------|------------|\n")
        
        for category, data in sorted(results['categories'].items()):
            percent = round(data['tested'] / data['with_from_pretrained'] * 100, 2) if data['with_from_pretrained'] > 0 else 0
            f.write(f"| {category} | {data['total']} | {data['with_from_pretrained']} | {data['tested']} | {percent}% |\n")
        
        f.write(f"\n")
        
        # Report missing tests
        if results['missing_tests']:
            f.write(f"## Missing Tests\n\n")
            f.write(f"The following model classes have a from_pretrained method but no proper test:\n\n")
            
            f.write(f"| Category | Model Class | Issue |\n")
            f.write(f"|----------|-------------|-------|\n")
            
            for category, cls_name, issue in sorted(results['missing_tests'], key=lambda x: (x[0], x[1])):
                f.write(f"| {category} | {cls_name} | {issue} |\n")
            
            f.write(f"\n")
        
        # Report partial tests
        if results['partial_tests']:
            f.write(f"## Partial Tests\n\n")
            f.write(f"The following model classes have tests that need improvement:\n\n")
            
            f.write(f"| Category | Model Class | Test File | Issues |\n")
            f.write(f"|----------|-------------|-----------|--------|\n")
            
            for category, cls_name, test_file, details in sorted(results['partial_tests'], key=lambda x: (x[0], x[1])):
                issues = []
                if not details.get('has_from_pretrained_method', False):
                    issues.append("No test_from_pretrained method")
                if not details.get('calls_from_pretrained', False):
                    issues.append("Doesn't call from_pretrained")
                if not details.get('validates_output', False):
                    issues.append("No output validation")
                if not details.get('handles_errors', False):
                    issues.append("No error handling")
                
                test_file_name = os.path.basename(test_file)
                issues_str = ", ".join(issues)
                f.write(f"| {category} | {cls_name} | {test_file_name} | {issues_str} |\n")
            
            f.write(f"\n")
        
        f.write(f"---\n\n")
        f.write(f"This report was generated using `validate_from_pretrained_coverage.py`.\n")
        f.write(f"To update this report, run:\n```bash\npython validate_from_pretrained_coverage.py\n```\n")
    
    logger.info(f"Validation report written to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Validate from_pretrained test coverage')
    parser.add_argument('-i', '--inventory', default=INVENTORY_FILE,
                        help=f'Path to model inventory JSON file (default: {INVENTORY_FILE})')
    parser.add_argument('-t', '--test-dir', default=TEST_DIR,
                        help=f'Directory containing test files (default: {TEST_DIR})')
    parser.add_argument('-o', '--output', default='from_pretrained_validation_report.md',
                        help='Output markdown report file path (default: from_pretrained_validation_report.md)')
    
    args = parser.parse_args()
    
    # Load the inventory
    inventory = load_inventory(args.inventory)
    
    # Analyze test coverage
    logger.info("Analyzing test coverage...")
    results = analyze_test_coverage(inventory, args.test_dir)
    
    # Create the report
    create_coverage_report(results, args.output)
    
    # Output summary to console
    logger.info(f"Coverage summary:")
    logger.info(f"  - Classes with from_pretrained: {results['total_with_from_pretrained']}")
    logger.info(f"  - Classes with proper tests: {results['total_tested']} ({results['percent_tested']}%)")
    logger.info(f"  - Classes with partial tests: {len(results['partial_tests'])}")
    logger.info(f"  - Classes missing tests: {len(results['missing_tests'])}")
    
    # Return success if coverage is 100%, failure otherwise
    return 0 if results['percent_tested'] == 100 else 1


if __name__ == '__main__':
    sys.exit(main())
