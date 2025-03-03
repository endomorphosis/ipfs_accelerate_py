"""
Hardware test templates module for IPFS Accelerate Python Framework.

This module provides templates for testing models on various hardware platforms.
"""

from .template_database import TemplateDatabase, get_template, store_template, list_templates

__all__ = ['TemplateDatabase', 'get_template', 'store_template', 'list_templates']