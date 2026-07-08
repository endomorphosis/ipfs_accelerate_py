#!/usr/bin/env python
"""
Test script for the template validator system.

This script demonstrates the functionality of the template validator
by validating hardware test templates and generating reports.
"""

import os
import sys
import argparse
from pathlib import Path

# Import template validator - skip database-related functions if not available
sys.path.append(str(Path(__file__).parent))
try:
    from template_validator import (
        validate_template_file,
        validate_template_directory,
        validate_template_from_db,
        generate_validation_report,
        store_validation_results
    )
    db_available = True
except ImportError as e:
    from template_validator import (
        validate_template_file,
        validate_template_directory,
        generate_validation_report
    )
    db_available = False
    print(f"Database functionality not available: {e}")
    
    # Define stub functions to avoid errors
    def validate_template_from_db(*args, **kwargs):
        return {"valid": False, "errors": ["Database functionality not available"]}
    
    def store_validation_results(results, output_file=None, store_in_db=False, db_path=None):
        if output_file:
            with open(output_file, 'w') as f:
                import json
                json.dump(results, f, indent=2)
            return True
        return False

def validate_bert_template():
    """Validate the BERT template and print detailed results."""
    template_path = str(Path(__file__).parent / "hardware_test_templates" / "template_bert.py")
    
    print(f"\nValidating BERT template: {template_path}\n")
    
    result = validate_template_file(template_path)
    
    print("\nValidation Results:")
    print(f"Valid: {'✅' if result['valid'] else '❌'}")
    print(f"Supported platforms: {', '.join(result.get('supported_platforms', []))}")
    
    if not result['valid']:
        print("\nErrors:")
        for error in result['errors']:
            print(f"  - {error}")
    
    print("\nDetailed Results by Validator:")
    for validator_name, validator_result in result.get('validators', {}).items():
        valid_symbol = "✅" if validator_result.get('valid', False) else "❌"
        print(f"{valid_symbol} {validator_name}:")
        for error in validator_result.get('errors', []):
            print(f"  - {error}")
    
    return result

def validate_all_templates():
    """Validate all hardware test templates and generate a report."""
    templates_dir = str(Path(__file__).parent / "hardware_test_templates")
    
    print(f"\nValidating all templates in: {templates_dir}\n")
    
    results = validate_template_directory(templates_dir)
    
    # Count valid and invalid templates
    valid_count = sum(1 for r in results.values() if r.get('valid', False))
    invalid_count = len(results) - valid_count
    
    print(f"\nValidation Results: {valid_count} valid, {invalid_count} invalid templates")
    
    # Generate platform support stats
    platform_counts = {
        'cuda': 0,
        'cpu': 0,
        'mps': 0,
        'rocm': 0,
        'openvino': 0,
        'webnn': 0,
        'webgpu': 0
    }
    
    for result in results.values():
        for platform in result.get('supported_platforms', []):
            if platform in platform_counts:
                platform_counts[platform] += 1
    
    print("\nHardware Platform Support:")
    for platform, count in platform_counts.items():
        percentage = count/len(results)*100 if results else 0
        print(f"  - {platform}: {count} templates ({percentage:.1f}%)")
    
    # Show details for invalid templates
    if invalid_count > 0:
        print("\nInvalid Templates:")
        for template_name, result in results.items():
            if not result.get('valid', True):
                print(f"  - {template_name}:")
                for error in result.get('errors', [])[:3]:  # Show first 3 errors
                    print(f"    - {error}")
                if len(result.get('errors', [])) > 3:
                    print(f"    - ... and {len(result.get('errors', [])) - 3} more errors")
    
    # Generate and save report
    report = generate_validation_report(results)
    report_path = str(Path(__file__).parent / "template_validation_report.md")
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\nValidation report saved to: {report_path}")
    
    # Store results
    results_path = str(Path(__file__).parent / "template_validation_results.json")
    store_validation_results(results, results_path)
    
    print(f"Validation results saved to: {results_path}")
    
    return results

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test template validation system")
    parser.add_argument("--all", action="store_true", help="Validate all templates")
    parser.add_argument("--bert", action="store_true", help="Validate BERT template")
    parser.add_argument("--template", type=str, help="Validate a specific template")
    args = parser.parse_args()
    
    if args.bert:
        validate_bert_template()
    elif args.template:
        template_path = args.template
        result = validate_template_file(template_path)
        
        print("\nValidation Results:")
        print(f"Valid: {'✅' if result['valid'] else '❌'}")
        print(f"Supported platforms: {', '.join(result.get('supported_platforms', []))}")
        
        if not result['valid']:
            print("\nErrors:")
            for error in result['errors']:
                print(f"  - {error}")
    elif args.all:
        validate_all_templates()
    else:
        # Default to validating BERT template
        validate_bert_template()

if __name__ == "__main__":
    main()