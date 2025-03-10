"""
Template Validator Package

This package provides tools for validating templates in generator scripts.
It includes:
- Template validator integration module
- Helper scripts for applying validation to generators
- Test script for verifying validator functionality
"""

from .template_validator_integration import (
    validate_template_for_generator,
    validate_template_file_for_generator,
    TemplateValidator
)

__all__ = [
    'validate_template_for_generator',
    'validate_template_file_for_generator',
    'TemplateValidator'
]