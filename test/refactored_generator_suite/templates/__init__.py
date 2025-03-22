#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Templates package for the refactored generator suite.
Provides access to all template implementations.
"""

import logging

# Import template classes
from .base import TemplateBase
from .encoder_only import EncoderOnlyTemplate
from .decoder_only import DecoderOnlyTemplate
from .encoder_decoder import EncoderDecoderTemplate
from .vision import VisionTemplate
from .vision_text import VisionTextTemplate
from .speech import SpeechTemplate

# Set up template registry
TEMPLATES = {
    "base": TemplateBase,
    "encoder-only": EncoderOnlyTemplate,
    "decoder-only": DecoderOnlyTemplate,
    "encoder-decoder": EncoderDecoderTemplate,
    "vision": VisionTemplate,
    "vision-text": VisionTextTemplate,
    "speech": SpeechTemplate,
}

# Set up basic logging
logging.basicConfig(level=logging.INFO)

# Package exports
__all__ = [
    'TemplateBase',
    'EncoderOnlyTemplate',
    'DecoderOnlyTemplate',
    'EncoderDecoderTemplate',
    'VisionTemplate',
    'VisionTextTemplate',
    'SpeechTemplate',
    'TEMPLATES',
    'get_template'
]

def get_template(template_name, config=None):
    """Get a template by name.
    
    Args:
        template_name: Name of the template.
        config: Optional configuration for the template.
        
    Returns:
        Template instance.
    """
    if template_name not in TEMPLATES:
        raise ValueError(f"Unknown template: {template_name}")
    
    template_class = TEMPLATES[template_name]
    return template_class(config)

def get_all_templates(config=None):
    """Get all available templates.
    
    Args:
        config: Optional configuration for the templates.
        
    Returns:
        Dictionary of template instances.
    """
    templates = {}
    for name, template_class in TEMPLATES.items():
        templates[name] = template_class(config)
    return templates