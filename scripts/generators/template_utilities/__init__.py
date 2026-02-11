"""Template utilities package

This package provides utilities for template rendering, placeholder management,
and template validation for the IPFS Accelerate Python Framework.
"""

from .placeholder_helpers import (
    get_standard_placeholders,
    detect_missing_placeholders,
    get_default_context,
    render_template,
    extract_placeholders,
    validate_placeholders,
    get_modality_for_model_type
)

__all__ = [
    'get_standard_placeholders',
    'detect_missing_placeholders',
    'get_default_context',
    'render_template',
    'extract_placeholders',
    'validate_placeholders',
    'get_modality_for_model_type'
]