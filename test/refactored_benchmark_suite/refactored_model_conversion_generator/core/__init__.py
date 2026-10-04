"""
Model Conversion Generator Core Package

This package provides core utilities for model format conversion
between different hardware backends.
"""

from .converter import ModelConverter, ConversionResult
from .registry import ModelConverterRegistry, register_converter

__all__ = [
    'ModelConverter',
    'ConversionResult',
    'ModelConverterRegistry',
    'register_converter'
]