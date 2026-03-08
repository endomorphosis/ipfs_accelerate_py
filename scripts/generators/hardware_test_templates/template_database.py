"""Compatibility shim for legacy generator imports.

Older generator utilities imported `hardware_test_templates.template_database`
from the generator workspace root. The canonical implementation now lives in
`template_database.py` alongside those scripts, so this shim re-exports that
surface without duplicating logic.
"""

from template_database import TemplateDatabase, get_template, list_templates

__all__ = ["TemplateDatabase", "get_template", "list_templates"]
